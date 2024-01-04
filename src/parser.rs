use std::fmt::Write;

use crate::lexer::*;

type Range = (usize, usize);

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode<'a> {
    Prog {
        body: Vec<AstNode<'a>>,
        range: Range,
    },
    FuncDecl {
        id: Option<Box<AstNode<'a>>>,
        params: Vec<AstNode<'a>>,
        body: Box<AstNode<'a>>,
        range: Range,
    },
    Ident {
        name: &'a str,
        range: Range,
    },
    StrLit {
        value: &'a str,
        format: bool,
        range: Range,
    },
    NumLit {
        value: f64,
        range: Range,
    },
}

impl<'a> AstNode<'a> {
    pub fn range(&self) -> Range {
        match self {
            AstNode::Prog { range, .. }
            | AstNode::FuncDecl { range, .. }
            | AstNode::Ident { range, .. }
            | AstNode::StrLit { range, .. }
            | AstNode::NumLit { range, .. } => *range,
        }
    }
}

type PResult = Result<AstNode<'static>, PError>;

#[derive(Debug)]
pub enum PError {
    ParseError,
    SyntaxError { expected: &'static [&'static str] },
}

/// When a parser fails, the lexer is reset to its state before the parser was called.
macro_rules! try_parse {
    ($self:ident: $parser:ident) => {{
        let before = $self.lexer.clone();
        match $self.$parser() {
            Ok(result) => Ok(result),
            Err(result) => {
                $self.lexer = before;
                Err(result)
            }
        }
    }};
}

/// Tries to match a parser one or more times, separated by another parser, returning a vector of
/// the results.
macro_rules! sep {
    ($self:ident: $parser:ident / $sep:ident) => {{
        let mut results = Vec::new();

        loop {
            let result = try_parse!($self: $parser);
            if let Ok(result) = result {
                results.push(result);
            } else {
                break;
            }

            if try_parse!($self: $sep).is_err() {
                break;
            }
        }

        results
    }};
}

/// Tries a list of parsers and returns the result of the first one that
/// succeeds, or an error if none succeed.
macro_rules! any {
    ($first_exp:literal $(, $rest_exp:literal)*; $self:ident: $first:ident $(| $rest:ident)*) => {
        {
            if let Ok(result) = try_parse!($self: $first) {
                Ok(result)
            } $(else if let Ok(result) = try_parse!($self: $rest) {
                Ok(result)
            })* else {
                Err(PError::SyntaxError {
                    expected: &[$first_exp $(, $rest_exp)*],
                })
            }
        }
    };
}

/// Matches a parser zero or more times, returning a vector of the results.
macro_rules! many {
    ($self:ident: $parser:ident) => {{
        let mut results = Vec::new();

        loop {
            let result = try_parse!($self: $parser);
            if let Ok(result) = result {
                results.push(result);
            } else {
                break;
            }
        }

        results
    }};
}

/// Matches a token, returning it if it succeeds, or an error if it fails.
macro_rules! token {
    ($name:literal; $self:ident: $($token:pat => $result:expr),*) => {{
        match $self.lexer.peek() {
            $($token => {
                $self.lexer.next();
                $result
            }),*,
            _ => Err(PError::SyntaxError {
                expected: &[$name],
            }),
        }
    }};
}

/// Tries to match a token, returning `Some(token)` if it succeeds, or `None` if it fails.
macro_rules! try_token {
    ($self:ident: $token:pat) => {{
        match $self.lexer.peek() {
            $token => Some($self.lexer.next()),
            _ => None,
        }
    }};
}

/// Expects a token and throws error if it fails.
macro_rules! expect_token {
    ($self:ident: $token:pat) => {{
        if let $token = $self.lexer.peek() {
            $self.lexer.next();
        } else {
            return Err(PError::ParseError);
        }
    }};
}

/// Matches a token zero or more times, returning a vector of the results.
macro_rules! many_token {
    ($self:ident: $token:pat) => {{
        let mut results = Vec::new();

        loop {
            match $self.lexer.peek() {
                $token => results.push($self.lexer.next()),
                _ => break,
            }
        }

        results
    }};
}

pub struct Parser {
    lexer: Lexer,
}

impl Parser {
    pub fn new(lexer: Lexer) -> Self {
        Parser { lexer }
    }

    pub fn parse(&mut self) -> PResult {
        let program = self.parse_program()?;

        if self.lexer.peek() != Token::EOF {
            return Err(PError::ParseError);
        }

        Ok(program)
    }

    pub fn display_error(&mut self, err: PError) -> Result<String, std::fmt::Error> {
        let mut msg = String::new();

        let next = self.lexer.peek();
        let span = next.span();
        let line = span.line;
        let column = span.column;
        let start = span.range.0;
        let end = span.range.1;

        match err {
            PError::ParseError => {
                write!(&mut msg, "Parsing Error @ {}:{}\n", line, column)?;
                self.display_error_line(&mut msg, line, column, start, end);

                write!(&mut msg, "Unexpected {}.", next.to_string())?;
            }
            PError::SyntaxError { expected } => {
                write!(&mut msg, "Syntax Error @ {}:{}\n", line, column)?;
                self.display_error_line(&mut msg, line, column, start, end);

                write!(&mut msg, "Expected one of:")?;
                for e in expected {
                    write!(&mut msg, "\n - {}", e)?;
                }
            }
        }

        Ok(msg)
    }

    fn display_error_line(
        &mut self,
        msg: &mut String,
        line: usize,
        column: usize,
        start: usize,
        end: usize,
    ) {
        let code = self.lexer.code_line(line);
        let marker = format!("{} | ", line);
        write!(msg, "{}{}\n", marker, code).unwrap();
        write!(msg, "{}", " ".repeat(column - 1 + marker.len())).unwrap();
        write!(msg, "{}\n", "^".repeat(end - start)).unwrap();
    }

    fn parse_newline(&mut self) -> Result<(), PError> {
        token! {
            "newline";
            self: Token::Newline { .. } => Ok(())
        }
    }

    fn parse_program(&mut self) -> PResult {
        many_token!(self: Token::Newline { .. });

        let body = sep!(self: parse_stmt / parse_newline);

        many_token!(self: Token::Newline { .. });

        let range = (0, body.last().map(|stmt| stmt.range().1).unwrap_or(0));

        Ok(AstNode::Prog { body, range })
    }

    fn parse_stmt(&mut self) -> PResult {
        any!(
            "function declaration", "expression";
            self: parse_func_decl | parse_expr
        )
    }

    fn parse_func_decl(&mut self) -> PResult {
        let anon_marker = try_token!(self: Token::Symbol { value: '\\', .. });

        let range_start;
        let id = match anon_marker {
            Some(Token::Symbol { span, .. }) => {
                range_start = span.range.0;
                None
            }
            None => {
                range_start = self.lexer.peek().span().range.0;
                Some(Box::new(self.parse_identifier()?))
            }
            _ => unreachable!(),
        };

        let params = many!(self: parse_identifier);

        expect_token!(self: Token::AssignmentArrow { .. });

        let body = Box::new(self.parse_expr()?);

        let range = (range_start, body.range().1);

        Ok(AstNode::FuncDecl {
            id,
            params,
            body,
            range,
        })
    }

    fn parse_expr(&mut self) -> PResult {
        any!(
            "identifier", "string", "number";
            self: parse_identifier | parse_string | parse_number
        )
    }

    fn parse_identifier(&mut self) -> PResult {
        token! {
            "identifier";
            self: Token::Identifier { name, span } => {
                Ok(AstNode::Ident {
                    name,
                    range: (span.range.0, span.range.1),
                })
            }
        }
    }

    fn parse_string(&mut self) -> PResult {
        let format = try_token!(self: Token::Symbol { value: '$', .. });

        token! {
            "string";
            self: Token::String { value, span } => {
                Ok(AstNode::StrLit {
                    value,
                    format: format.is_some(),
                    range: (span.range.0, span.range.1),
                })
            }
        }
    }

    fn parse_number(&mut self) -> PResult {
        token! {
            "number";
            self: Token::Number { value: whole, span: w_span } => {
                if try_token!(self: Token::Symbol { value: '.', .. }).is_none() {
                    return Ok(AstNode::NumLit {
                        value: whole as f64,
                        range: (w_span.range.0, w_span.range.1),
                    });
                }

                token! {
                    "number";
                    self: Token::Number { value: fraction, span: f_span } => {
                        let number = format!("{}.{}", whole, fraction);
                        Ok(AstNode::NumLit {
                            value: number.parse().unwrap(),
                            range: (w_span.range.0, f_span.range.1),
                        })
                    }
                }
            }
        }
    }
}
