use std::fmt::Write;

use crate::lexer::*;

type Range = (usize, usize);

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode<'a> {
    Prog {
        body: Vec<AstNode<'a>>,
        range: Range,
    },
    ExprStmt {
        expr: Box<AstNode<'a>>,
        range: Range,
    },
    FuncDecl {
        id: Option<Box<AstNode<'a>>>,
        params: Vec<AstNode<'a>>,
        body: Box<AstNode<'a>>,
        range: Range,
    },
    FuncCall {
        callee: Box<AstNode<'a>>,
        arg: Option<Box<AstNode<'a>>>,
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
            | AstNode::ExprStmt { range, .. }
            | AstNode::FuncDecl { range, .. }
            | AstNode::FuncCall { range, .. }
            | AstNode::Ident { range, .. }
            | AstNode::StrLit { range, .. }
            | AstNode::NumLit { range, .. } => *range,
        }
    }
}

type PResult = Result<AstNode<'static>, PError>;

#[derive(Debug, Clone)]
pub enum PError {
    ParseError,
    SyntaxError { expected: &'static [&'static str] },

    Cut(Box<PError>, Span),
}

/// Transforms a recoverable error into an unrecoverable one.
macro_rules! cut {
    ($self:ident: $parser:ident) => {
        match $self.$parser() {
            Ok(success) => Ok(success),
            Err(err) => Err(PError::Cut(Box::new(err), $self.lexer.peek().span())),
        }
    };
}

/// Tries to match a parser or token. Returns `Option<T>`.
///
/// After failing, resets the cursor to its position before the parser was called.
macro_rules! opt {
    ($self:ident: $parser:ident) => {{
        let cursor = $self.lexer.cursor();
        match $self.$parser() {
            Ok(success) => Some(success),
            Err(err) => {
                if let PError::Cut(_, _) = err {
                    // Unrecoverable error, propagate it
                    $self.set_last_error(Some(err), true);
                    return Err(PError::ParseError);
                }

                $self.lexer.set_cursor(cursor);
                $self.set_last_error(Some(err), false);
                None
            }
        }
    }};
    ($self:ident: $token:pat) => {
        match $self.lexer.peek() {
            $token => Some($self.lexer.next()),
            _ => None,
        }
    };
}

/// Tries a list of parsers and stops on the first success.
macro_rules! any {
    ($self:ident: $first:ident, $first_exp:literal $(| $rest:ident, $rest_exp:literal )*) => {
        if let Some(success) = opt!($self: $first) {
            Ok(success)
        } $(else if let Some(success) = opt!($self: $rest) {
            Ok(success)
        })* else {
            Err(PError::SyntaxError {
                expected: &[$first_exp $(, $rest_exp)*],
            })
        }
    };
}

/// Matches a parser `n` times, separated by another parser. On success, returns `Vec<T>`.
macro_rules! sep {
    ($n:expr, $self:ident: $parser:ident / $sep:ident) => {{
        let mut results = Vec::new();
        let mut error = None;

        loop {
            let cursor = $self.lexer.cursor();
            match $self.$parser() {
                Ok(success) => results.push(success),
                Err(err) => {
                    $self.lexer.set_cursor(cursor);
                    $self.set_last_error(Some(err.clone()), false);
                    error = Some(err);
                    break;
                }
            }

            if opt!($self: $sep).is_none() {
                break;
            }
        }

        if !$n.contains(&results.len()) {
            return Err(error.unwrap());
        }

        Ok(results)
    }};
}

/// Matches a parser `n` times. On success, returns `Vec<T>`.
macro_rules! many {
    ($n:expr, $self:ident: $parser:ident) => {{
        let mut results = Vec::new();

        loop {
            let cursor = $self.lexer.cursor();
            match $self.$parser() {
                Ok(success) => results.push(success),
                Err(err) => {
                    $self.lexer.set_cursor(cursor);
                    $self.set_last_error(Some(err), false);
                    break;
                }
            }
        }

        if !$n.contains(&results.len()) {
            return Err(PError::ParseError);
        }

        Ok(results)
    }};
}

/// Matches a token.
macro_rules! token {
    ($self:ident: $($token:pat, $name:literal => $result:expr),*) => {
        match $self.lexer.peek() {
            $($token => {
                $self.lexer.next();
                Ok($result)
            }),*,
            _ => Err(PError::SyntaxError {
                expected: &[$($name),*],
            }),
        }
    };
}

pub struct Parser {
    lexer: Lexer,
    cut: bool,
    last_error: Option<PError>,
}

impl Parser {
    pub fn new(lexer: Lexer) -> Self {
        Parser {
            lexer,
            cut: false,
            last_error: None,
        }
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
        let mut span = next.span();
        let mut err = err;

        if self.cut {
            if let Some(PError::Cut(inner_err, inner_span)) = self.last_error.take() {
                err = *inner_err;
                span = inner_span;
            }
        }

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
            _ => unreachable!("unrecoverable error"),
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

    fn set_last_error(&mut self, err: Option<PError>, cut: bool) {
        if !self.cut || self.last_error.is_none() {
            self.last_error = err;
            self.cut = cut;
        }
    }

    fn parse_newline(&mut self) -> Result<(), PError> {
        token!(self: Token::Newline { .. }, "newline" => ())
    }

    fn parse_program(&mut self) -> PResult {
        many!(0.., self: parse_newline)?;

        let body = sep!(1.., self: parse_stmt / parse_newline)?;

        many!(0.., self: parse_newline)?;

        let range = (0, body.last().map(|stmt| stmt.range().1).unwrap_or(0));

        Ok(AstNode::Prog { body, range })
    }

    fn parse_stmt(&mut self) -> PResult {
        any! {
            self: parse_func_decl, "function declaration"
                | parse_expr_stmt, "expression"
        }
    }

    fn parse_expr_stmt(&mut self) -> PResult {
        let expr = self.parse_func_call()?;
        let range = expr.range();

        Ok(AstNode::ExprStmt {
            expr: Box::new(expr),
            range,
        })
    }

    fn parse_func_decl(&mut self) -> PResult {
        let anon_marker = opt!(self: Token::Symbol { value: '\\', .. });

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

        let params = many!(0.., self: parse_identifier)?;

        token!(self: Token::AssignmentArrow { .. }, "->" => { })?;

        let body = cut!(self: parse_expr)?;

        let range = (range_start, body.range().1);

        Ok(AstNode::FuncDecl {
            id,
            params,
            body: Box::new(body),
            range,
        })
    }

    fn parse_expr(&mut self) -> PResult {
        any! {
            self: parse_identifier, "identifier"
                | parse_string, "string"
                | parse_number, "number"
        }
    }

    fn parse_func_call(&mut self) -> PResult {
        let id = self.parse_identifier()?;
        let range = id.range();

        let mut current = id;

        loop {
            let result = opt!(self: parse_expr);
            if result.is_none() {
                break;
            }

            let arg = result.unwrap();
            let range = (current.range().0, arg.range().1);

            current = AstNode::FuncCall {
                callee: Box::new(current),
                arg: Some(Box::new(arg.clone())),
                range,
            };
        }

        let call = match current {
            AstNode::Ident { name, .. } => AstNode::FuncCall {
                callee: Box::new(AstNode::Ident { name, range }),
                arg: None,
                range,
            },
            _ => current,
        };

        Ok(call)
    }

    fn parse_identifier(&mut self) -> PResult {
        token! {
            self: Token::Identifier { name, span }, "identifier" => {
                AstNode::Ident {
                    name,
                    range: span.range,
                }
            }
        }
    }

    fn parse_string(&mut self) -> PResult {
        let format_marker = opt!(self: Token::Symbol { value: '$', .. });

        let (value, span) = if format_marker.is_some() {
            cut!(self: parse_string_part)?
        } else {
            self.parse_string_part()?
        };

        Ok(AstNode::StrLit {
            value,
            format: format_marker.is_some(),
            range: span.range,
        })
    }

    fn parse_string_part(&mut self) -> Result<(&'static str, Span), PError> {
        token!(self: Token::String { value, span }, "string" => (value, span))
    }

    fn parse_number(&mut self) -> PResult {
        let whole = self.parse_number_part()?;

        if opt!(self: Token::Symbol { value: '.', .. }).is_none() {
            return Ok(AstNode::NumLit {
                value: whole as f64,
                range: (0, 0),
            });
        }

        let fraction = cut!(self: parse_number_part)?;

        Ok(AstNode::NumLit {
            value: format!("{}.{}", whole, fraction).parse::<f64>().unwrap(),
            range: (0, 0),
        })
    }

    fn parse_number_part(&mut self) -> Result<u64, PError> {
        token!(self: Token::Number { value, .. }, "number" => value)
    }
}
