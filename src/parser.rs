use codespan_reporting::diagnostic::{Diagnostic, Label};
use ecow::{EcoString, EcoVec};

use crate::lexer::*;

static UNDERLINE: &str = "_";

type Range = (usize, usize);

#[derive(Debug, Clone, PartialEq)]
pub enum CasePatternKind {
    Any,
    Ident,
    Literal,
    // Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    Prog {
        body: EcoVec<AstNode>,
        range: Range,
    },
    ExprStmt {
        expr: Box<AstNode>,
        range: Range,
    },
    Block {
        body: EcoVec<AstNode>,
        range: Range,
    },
    FuncDecl {
        id: Box<AstNode>,
        is_const: bool,
        params: EcoVec<AstNode>,
        body: Box<AstNode>,
        range: Range,
    },
    FuncRef {
        id: Box<AstNode>,
        range: Range,
    },
    Lambda {
        params: EcoVec<AstNode>,
        body: Box<AstNode>,
        range: Range,
    },
    Case {
        expr: Box<AstNode>,
        cases: EcoVec<AstNode>,
        range: Range,
    },
    CaseBranch {
        pattern: Box<AstNode>,
        body: Box<AstNode>,
        range: Range,
    },
    CasePattern {
        kind: CasePatternKind,
        expr: Box<AstNode>,
        range: Range,
    },
    FuncCall {
        callee: Box<AstNode>,
        args: EcoVec<AstNode>,
        range: Range,
    },
    IndexingOp {
        expr: Box<AstNode>,
        index: Box<AstNode>,
        range: Range,
    },
    ConcatOp {
        left: Box<AstNode>,
        right: Box<AstNode>,
        range: Range,
    },
    IterationOp {
        id: Box<AstNode>,
        expr: Box<AstNode>,
        body: Box<AstNode>,
        range: Range,
    },
    Ident {
        name: EcoString,
        range: Range,
    },
    BoolLit {
        value: bool,
        range: Range,
    },
    StrLit {
        value: EcoString,
        is_format: bool,
        range: Range,
    },
    Regex {
        value: EcoString,
        is_case_insensitive: bool,
        is_global: bool,
        is_multiline: bool,
        is_dotall: bool,
        // is_unicode: bool,
        is_sticky: bool,
        range: Range,
    },
    NumLit {
        value: f64,
        range: Range,
    },
    List {
        items: EcoVec<AstNode>,
        range: Range,
    },
    Record {
        keys: EcoVec<AstNode>,
        values: EcoVec<AstNode>,
        range: Range,
    },
    Range {
        mode: RangeMode,
        start: Box<AstNode>,
        end: Box<AstNode>,
        range: Range,
    },
}

impl AstNode {
    pub fn range(&self) -> Range {
        match self {
            AstNode::Prog { range, .. }
            | AstNode::ExprStmt { range, .. }
            | AstNode::Block { range, .. }
            | AstNode::FuncDecl { range, .. }
            | AstNode::FuncRef { range, .. }
            | AstNode::Lambda { range, .. }
            | AstNode::Case { range, .. }
            | AstNode::CaseBranch { range, .. }
            | AstNode::CasePattern { range, .. }
            | AstNode::FuncCall { range, .. }
            | AstNode::IndexingOp { range, .. }
            | AstNode::ConcatOp { range, .. }
            | AstNode::IterationOp { range, .. }
            | AstNode::Ident { range, .. }
            | AstNode::BoolLit { range, .. }
            | AstNode::StrLit { range, .. }
            | AstNode::Regex { range, .. }
            | AstNode::NumLit { range, .. }
            | AstNode::List { range, .. }
            | AstNode::Record { range, .. }
            | AstNode::Range { range, .. } => *range,
        }
    }
}

type PResult = Result<AstNode, ParseError>;

#[derive(Debug, Clone)]
pub enum ParseError {
    Fail,
    Unexpected { expected: &'static [&'static str] },
    Forbidden(&'static str),
    InvalidIndent(usize),
    InvalidDedent(usize),

    Cut(Box<ParseError>, Token),
}

/// Runs a parser, propagating unrecoverable errors. (For internal use by macros.)
macro_rules! parse {
    ($self:ident: $parser:ident) => {{
        let cursor = $self.lexer.cursor();
        match $self.$parser() {
            Ok(success) => Ok(success),
            Err(err) => {
                if let ParseError::Cut(_, _) = err {
                    // Unrecoverable error
                    $self.set_cut_error(Some(err));
                    return Err(ParseError::Fail);
                }

                $self.lexer.set_cursor(cursor);
                Err(err)
            }
        }
    }};
}

/// Transforms a recoverable error into an unrecoverable one.
macro_rules! cut {
    ($self:ident: $parser:ident) => {
        match $self.$parser() {
            Ok(success) => Ok(success),
            Err(ParseError::Cut(inner, span)) => Err(ParseError::Cut(inner, span)),
            Err(err) => Err(ParseError::Cut(Box::new(err), $self.lexer.peek())),
        }
    };
}

/// Matches the next token against a pattern. On success, returns `Token`.
macro_rules! token {
    ($name:literal; $self:ident: $token:pat) => {
        match $self.lexer.peek() {
            $token => Ok($self.lexer.next()),
            _ => Err(ParseError::Unexpected { expected: &[$name] }),
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
            Err(_) => {
                $self.lexer.set_cursor(cursor);
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

/// Similar to `opt!`, but does not consume the result.
macro_rules! peek {
    ($self:ident: $parser:ident) => {{
        let cursor = $self.lexer.cursor();

        let result = $self.$parser();

        $self.lexer.set_cursor(cursor);

        result
    }};
    ($n:expr, $self:ident: $token:pat) => {
        match $self.lexer.peek_nth($n) {
            $token => Some($self.lexer.peek()),
            _ => None,
        }
    };
}

/// Tries a list of parsers and stops on the first success.
macro_rules! any {
    ($self:ident: $first:ident, $first_exp:literal $(| $rest:ident, $rest_exp:literal)*) => {{
        if let Ok(success) = parse!($self: $first) {
            Ok(success)
        } $(else if let Ok(success) = parse!($self: $rest) {
            Ok(success)
        })* else {
            Err(ParseError::Unexpected {
                expected: &[$first_exp $(, $rest_exp)*],
            })
        }
    }};
}

/// Matches a parser or token `n` times, separated by another parser or token. On success, returns `Vec<T>`.
macro_rules! sep {
    ($n:expr, $self:ident: $parser:ident / $sep:ident) => {{
        let mut results = Vec::new();

        loop {
            let result = parse!($self: $parser);
            if result.is_err() {
                break;
            }

            results.push(result.unwrap());

            if opt!($self: $sep).is_none() {
                break;
            }
        }

        if !$n.contains(&results.len()) {
            return Err(ParseError::Fail);
        }

        Ok(results)
    }};
    ($n:expr, $self:ident: $parser:ident / $sep:pat) => {{
        let mut results = Vec::new();

        loop {
            let result = parse!($self: $parser);
            if result.is_err() {
                break;
            }

            results.push(result.unwrap());

            match $self.lexer.peek() {
                $sep => {
                    $self.lexer.next();
                }
                _ => break
            }
        }

        if !$n.contains(&results.len()) {
            return Err(ParseError::Fail);
        }

        Ok(results)
    }};
}

/// Matches a parser or token `n` times. On success, returns `Vec<T>`.
macro_rules! many {
    ($n:expr, $self:ident: $parser:ident) => {{
        let mut results = Vec::new();

        loop {
            let result = parse!($self: $parser);
            if result.is_err() {
                break;
            }

            results.push(result.unwrap());
        }

        if !$n.contains(&results.len()) {
            return Err(ParseError::Fail);
        }

        Ok(results)
    }};
    ($n:expr, $self:ident: $token:pat) => {{
        let mut results = Vec::new();

        loop {
            match $self.lexer.peek() {
                $token => {
                    results.push($self.lexer.next());
                }
                _ => break,
            }
        }

        if !$n.contains(&results.len()) {
            return Err(ParseError::Fail);
        }

        Ok(results)
    }};
}

/// Matches the next token against a pattern and maps it to a result.
macro_rules! map {
    ($self:ident: $($token:pat, $name:literal => $result:expr),*) => {
        match $self.lexer.peek() {
            $($token => {
                $self.lexer.next();
                Ok($result)
            }),*,
            _ => Err(ParseError::Unexpected {
                expected: &[$($name),*],
            }),
        }
    };
}

pub struct Parser {
    lexer: Lexer,
    cut_error: Option<ParseError>,
    indent_stack: Vec<usize>,
    indent_size: usize,
}

impl Parser {
    pub fn new(lexer: Lexer) -> Self {
        Parser {
            lexer,
            cut_error: None,
            indent_stack: Vec::new(),
            indent_size: 0,
        }
    }

    pub fn parse(&mut self) -> PResult {
        let program = self.parse_program()?;

        match self.lexer.peek() {
            Token::EOF { .. } => {}
            _ => {
                return Err(ParseError::Fail);
            }
        }

        Ok(program)
    }

    pub fn error_to_diagnostic(&mut self, error: ParseError) -> Diagnostic<usize> {
        let tok;
        let span;
        let err;

        match self.cut_error.take() {
            Some(ParseError::Cut(inner_err, inner_tok)) => {
                span = inner_tok.span();
                tok = inner_tok;
                err = *inner_err;
            }
            _ => {
                tok = self.lexer.peek();
                span = tok.span();
                err = error;
            }
        }

        match tok {
            Token::EOF { span } => {
                return Diagnostic::error()
                    .with_message("Parsing Error: Unexpected end of file.")
                    .with_labels(vec![Label::primary(0, span.range.0..span.range.1)]);
            }
            Token::Indent { size, span } | Token::Dedent { size, span } => match err {
                ParseError::Unexpected { .. } | ParseError::Fail => {
                    return Diagnostic::error()
                        .with_code("parse")
                        .with_message("Wrong indentation.")
                        .with_labels(vec![Label::primary(0, span.range.0..span.range.1)])
                        .with_notes(vec![format!(
                            "Expected indent of size {}, found {}.",
                            self.indent_stack.last().unwrap_or(&0),
                            size
                        )]);
                }
                _ => {}
            },
            _ => {}
        }

        match err {
            ParseError::Fail => Diagnostic::error()
                .with_code("parse")
                .with_message(format!("Unexpected {}.", tok.to_string()))
                .with_labels(vec![Label::primary(0, span.range.0..span.range.1)])
                .with_notes(vec!["Note: This is probably a syntax error.".to_string()]),
            ParseError::Unexpected { expected } => {
                let expected = {
                    if expected.len() == 1 {
                        let article = match expected[0].chars().next().unwrap() {
                            'a' | 'e' | 'i' | 'o' | 'u' => "an",
                            _ => "a",
                        };
                        format!("Expected {} {}.", article, expected[0])
                    } else {
                        match expected.len() {
                            1 => format!("Expected {}.", expected[0]),
                            2 => format!("Expected one of: {} or {}.", expected[0], expected[1]),
                            _ => {
                                let mut expected = expected.to_vec();
                                let last = expected.pop().unwrap();
                                format!("Expected one of: {}, or {}.", expected.join(", "), last)
                            }
                        }
                    }
                };

                Diagnostic::error()
                    .with_code("parse")
                    .with_message(format!("Unexpected {}.", tok.to_string()))
                    .with_labels(vec![Label::primary(0, span.range.0..span.range.1)])
                    .with_notes(vec![expected])
            }
            ParseError::Forbidden(reason) => Diagnostic::error()
                .with_code("parse")
                .with_message(reason)
                .with_labels(vec![Label::primary(0, span.range.0..span.range.1)]),
            ParseError::InvalidIndent(size) => Diagnostic::error()
                .with_code("parse")
                .with_message("Wrong indentation.")
                .with_labels(vec![Label::primary(0, span.range.0..span.range.1)])
                .with_notes(vec![format!(
                    "Expected indent of size {}, found {}.",
                    self.indent_stack.last().unwrap_or(&0) + self.indent_size,
                    size
                )]),
            ParseError::InvalidDedent(size) => Diagnostic::error()
                .with_code("parse")
                .with_message("Wrong indentation.")
                .with_labels(vec![Label::primary(0, span.range.0..span.range.1)])
                .with_notes(vec![format!(
                    "Expected indent of size {}, found {}.",
                    self.indent_stack.last().unwrap_or(&0),
                    size
                )]),
            _ => unreachable!("unrecoverable error"),
        }
    }

    fn set_cut_error(&mut self, err: Option<ParseError>) {
        if self.cut_error.is_none() {
            self.cut_error = err;
        }
    }

    /// Transforms a recoverable error into an unrecoverable one.
    fn cut<T>(&mut self, result: Result<T, ParseError>) -> Result<T, ParseError> {
        match result {
            Ok(success) => Ok(success),
            Err(ParseError::Cut(inner, span)) => Err(ParseError::Cut(inner, span)),
            Err(err) => Err(ParseError::Cut(Box::new(err), self.lexer.peek())),
        }
    }

    fn ignore_nl(&mut self) -> Result<Vec<Token>, ParseError> {
        many!(0.., self: Token::Newline { .. })
    }

    fn ignore_nl_and_ws(&mut self) -> Result<Vec<Token>, ParseError> {
        many!(0.., self: Token::Newline { .. } | Token::Indent { .. } | Token::Dedent { .. })
    }

    fn parse_indent(&mut self) -> Result<(), ParseError> {
        let last_indent = self.indent_stack.last().unwrap_or(&0);

        let indent = token!("indent"; self: Token::Indent { .. })?;
        if let Token::Indent { size, .. } = indent {
            if self.indent_size == 0 {
                self.indent_size = size;
            }

            if size == *last_indent + self.indent_size {
                self.indent_stack.push(size);
                Ok(())
            } else {
                Err(ParseError::Cut(
                    Box::new(ParseError::InvalidIndent(size)),
                    indent,
                ))
            }
        } else {
            Err(ParseError::Unexpected {
                expected: &["indent"],
            })
        }
    }

    fn parse_dedent(&mut self) -> Result<(), ParseError> {
        let last_indent = self.indent_stack.last().unwrap_or(&0);

        let dedent = token!("dedent"; self: Token::Dedent { .. })?;
        if let Token::Dedent { size, .. } = dedent {
            if size == *last_indent {
                self.indent_stack.pop();
                Ok(())
            } else {
                Err(ParseError::Cut(
                    Box::new(ParseError::InvalidDedent(size)),
                    dedent,
                ))
            }
        } else {
            Err(ParseError::Unexpected {
                expected: &["dedent"],
            })
        }
    }

    fn parse_program(&mut self) -> PResult {
        self.ignore_nl()?;

        let body = sep!(0.., self: parse_stmt / Token::Newline { .. })?;

        self.ignore_nl()?;

        let range = (0, body.last().map(|stmt| stmt.range().1).unwrap_or(0));

        Ok(AstNode::Prog {
            body: body.into(),
            range,
        })
    }

    fn parse_stmt(&mut self) -> PResult {
        any! {
            self: parse_func_decl, "function declaration"
                | parse_case, "case statement"
                | parse_expr_stmt, "expression"
        }
    }

    fn parse_expr_stmt(&mut self) -> PResult {
        let expr = any! {
            self: parse_indexing, "indexing"
                | parse_concat, "concatenation"
                | parse_iteration_op, "iterator"
                | parse_func_call, "function call"
                | parse_expr, "expression"
        }?;
        let range = expr.range();

        Ok(AstNode::ExprStmt {
            expr: Box::new(expr),
            range,
        })
    }

    fn parse_block(&mut self) -> PResult {
        self.parse_indent()?;

        let mut body = EcoVec::new();

        loop {
            body.push(cut!(self: parse_block_stmt)?);

            self.ignore_nl()?;

            if peek!(0, self: Token::Dedent { .. }).is_some() {
                break;
            }
        }

        cut!(self: parse_dedent)?;

        if body.is_empty() {
            return Err(ParseError::Unexpected {
                expected: &["block statement"],
            });
        }

        let range = (
            body.first().map(|stmt| stmt.range().0).unwrap_or(0),
            body.last().map(|stmt| stmt.range().1).unwrap_or(0),
        );

        Ok(AstNode::Block { body, range })
    }

    fn parse_block_stmt(&mut self) -> PResult {
        any! {
            self: parse_func_decl, "function declaration"
                | parse_case, "case statement"
                | parse_indexing, "indexing"
                | parse_concat, "concatenation"
                | parse_iteration_op, "iterator"
                | parse_func_call, "function call"
                | parse_expr, "expression"
        }
    }

    fn parse_func_decl(&mut self) -> PResult {
        let const_marker = opt!(self: Token::Keyword { name: Keyword::Const, .. });

        let anon_marker = peek!(0, self: Token::Symbol { value: '!', .. });

        if anon_marker.is_some() {
            if const_marker.is_some() {
                let org_span = const_marker.unwrap().span();
                let culprit = Token::Keyword {
                    name: Keyword::Const,
                    span: Span {
                        line: org_span.line,
                        column: org_span.column,
                        range: (org_span.range.0, anon_marker.unwrap().span().range.1),
                    },
                };

                return Err(ParseError::Cut(
                    Box::new(ParseError::Forbidden(
                        "An anonymous function cannot be constant.",
                    )),
                    culprit,
                ));
            } else {
                return self.parse_lambda();
            }
        }

        let range_start = const_marker
            .as_ref()
            .map(|tok| tok.span().range.0)
            .unwrap_or_else(|| self.lexer.peek().span().range.0);

        let id = self.parse_identifier()?;

        let params = many!(0.., self: parse_identifier)?;

        if const_marker.is_some() && !params.is_empty() {
            let org_span = const_marker.unwrap().span();
            let culprit = Token::Keyword {
                name: Keyword::Const,
                span: Span {
                    line: org_span.line,
                    column: org_span.column,
                    range: (org_span.range.0, params.last().unwrap().range().1),
                },
            };

            return Err(ParseError::Cut(
                Box::new(ParseError::Forbidden(
                    "A constant function cannot have parameters.",
                )),
                culprit,
            ));
        }

        token!("`->`"; self: Token::AssignmentArrow { .. })?;

        let body = cut!(self: parse_func_block)?;

        let range = (range_start, body.range().1);

        Ok(AstNode::FuncDecl {
            id: Box::new(id),
            is_const: const_marker.is_some(),
            params: params.into(),
            body: Box::new(body),
            range,
        })
    }

    fn parse_lambda(&mut self) -> PResult {
        let anon_marker = token!("`!`"; self: Token::Symbol { value: '!', .. })?;

        let params = many!(0.., self: parse_identifier)?;

        token!("`->`"; self: Token::AssignmentArrow { .. })?;

        let body = cut!(self: parse_func_block)?;

        let range = (anon_marker.span().range.0, body.range().1);

        Ok(AstNode::Lambda {
            params: params.into(),
            body: Box::new(body),
            range,
        })
    }

    fn parse_func_block(&mut self) -> PResult {
        any! {
            self: parse_block, "indented block"
                | parse_case, "case statement"
                | parse_lambda, "lambda"
                | parse_indexing, "indexing"
                | parse_concat, "concatenation"
                | parse_func_decl, "function declaration"
                | parse_iteration_op, "iterator"
                | parse_func_call, "function call"
                | parse_expr, "expression"
        }
    }

    fn parse_case(&mut self) -> PResult {
        let case = token!("`case`"; self: Token::Keyword { name: Keyword::Case, .. })?;

        let expr = cut!(self: parse_expr)?;

        let of = token!("`of`"; self: Token::Keyword { name: Keyword::Of, .. });
        self.cut(of)?;

        cut!(self: parse_indent)?;

        let cases = cut!(self: parse_case_cases)?;

        cut!(self: parse_dedent)?;

        let range = (case.span().range.0, cases.last().unwrap().range().1);

        Ok(AstNode::Case {
            expr: Box::new(expr),
            cases: cases.into(),
            range,
        })
    }

    fn parse_case_cases(&mut self) -> Result<Vec<AstNode>, ParseError> {
        let mut cases = Vec::new();

        loop {
            cases.push(cut!(self: parse_case_branch)?);

            if peek!(0, self: Token::Dedent { .. }).is_some() {
                break;
            }

            if opt!(self: Token::Newline { .. }).is_none() {
                return Err(ParseError::Unexpected {
                    expected: &["newline", "dedent"],
                });
            }
        }

        if cases.is_empty() {
            return Err(ParseError::Unexpected {
                expected: &["case branch"],
            });
        }

        Ok(cases)
    }

    fn parse_case_branch(&mut self) -> PResult {
        let pattern = cut!(self: parse_case_pattern)?;

        let arrow = token!("`=>`"; self: Token::MatchArrow { .. });
        self.cut(arrow)?;

        let body = cut!(self: parse_case_body)?;

        let range = (pattern.range().0, body.range().1);

        Ok(AstNode::CaseBranch {
            pattern: Box::new(pattern),
            body: Box::new(body),
            range,
        })
    }

    fn parse_case_pattern(&mut self) -> PResult {
        // TODO:
        // - [ ] Implement `or` case pattern
        // - [ ] Implement `range` case pattern
        // - [ ] Implement `regex` case pattern
        // - [ ] Implement `list` case pattern
        // - [ ] Implement `record` case pattern

        let expr = cut!(self: parse_case_pattern_expr)?;
        match expr {
            AstNode::Range { .. } => todo!("range case pattern"),
            AstNode::Ident {
                ref name, range, ..
            } => {
                let kind = if name == UNDERLINE {
                    CasePatternKind::Any
                } else {
                    CasePatternKind::Ident
                };

                Ok(AstNode::CasePattern {
                    kind,
                    expr: Box::new(expr),
                    range,
                })
            }
            AstNode::BoolLit { range, .. } => Ok(AstNode::CasePattern {
                kind: CasePatternKind::Literal,
                expr: Box::new(expr),
                range,
            }),
            AstNode::StrLit { range, .. } => Ok(AstNode::CasePattern {
                kind: CasePatternKind::Literal,
                expr: Box::new(expr),
                range,
            }),
            AstNode::Regex { .. } => todo!("regex case pattern"),
            AstNode::NumLit { range, .. } => Ok(AstNode::CasePattern {
                kind: CasePatternKind::Literal,
                expr: Box::new(expr),
                range,
            }),
            AstNode::List { .. } => todo!("list case pattern"),
            AstNode::Record { .. } => todo!("record case pattern"),
            _ => unreachable!(),
        }
    }

    fn parse_case_body(&mut self) -> PResult {
        any! {
            self: parse_block, "indented block"
                | parse_case, "case statement"
                | parse_concat, "concatenation"
                | parse_indexing, "indexing"
                | parse_func_call, "function call"
                | parse_expr, "expression"
        }
    }

    fn parse_case_pattern_expr(&mut self) -> PResult {
        any! {
            self: parse_range, "range"
                | parse_identifier, "identifier"
                | parse_bool, "boolean"
                | parse_string, "string"
                | parse_regex, "regular expression"
                | parse_number, "number"
                | parse_list, "list"
                | parse_record, "record"
        }
    }

    // fn parse_case_pattern_or(&mut self) -> Result<Vec<AstNode>, ParseError> {
    //     let mut items = Vec::new();

    //     loop {
    //         let item = any! {
    //             self: parse_identifier, "identifier"
    //                 | parse_string, "string"
    //                 | parse_number, "number"
    //         };

    //         items.push(self.cut(item)?);

    //         if opt!(self: Token::Symbol { value: '|', .. }).is_none() {
    //             break;
    //         }
    //     }

    //     if items.is_empty() {
    //         return Err(ParseError::Unexpected {
    //             expected: &["identifier", "string", "number"],
    //         });
    //     }

    //     Ok(items)
    // }

    fn parse_expr(&mut self) -> PResult {
        any! {
            self: parse_range, "range"
                | parse_indexing, "indexing"
                | parse_concat, "concatenation"
                | parse_identifier, "identifier"
                | parse_bool, "boolean"
                | parse_string, "string"
                | parse_number, "number"
                | parse_list, "list"
                | parse_record, "record"
                | parse_par_expr, "expression"
                | parse_func_ref, "function reference"
                | parse_lambda, "lambda"
        }
    }

    fn parse_func_call(&mut self) -> PResult {
        let id = any! {
            self: parse_identifier, "identifier"
                | parse_func_ref, "function reference"
                | parse_par_expr, "expression"
                | parse_par_func_call, "function call"
        }?;

        let args = many!(0.., self: parse_expr)?;

        let range_start = id.range();
        let range = (
            range_start.0,
            args.last()
                .map(|arg| arg.range().1)
                .unwrap_or(range_start.1),
        );

        Ok(AstNode::FuncCall {
            callee: Box::new(id),
            args: args.into(),
            range,
        })
    }

    fn parse_par_func_call(&mut self) -> PResult {
        let start = token!("`(`"; self: Token::Symbol { value: '(', .. })?;

        self.ignore_nl_and_ws()?;

        let result = cut!(self: parse_func_call)?;

        self.ignore_nl_and_ws()?;

        let end = token!("`)`"; self: Token::Symbol { value: ')', .. });
        let end = self.cut(end)?;

        let range = (start.span().range.0, end.span().range.1);

        let result = match result {
            AstNode::FuncCall { callee, args, .. } => AstNode::FuncCall {
                callee,
                args,
                range,
            },
            _ => unreachable!(),
        };

        Ok(result)
    }

    fn parse_indexing(&mut self) -> PResult {
        let expr = any! {
            self: parse_range, "range"
                | parse_identifier, "identifier"
                | parse_bool, "boolean"
                | parse_string, "string"
                | parse_number, "number"
                | parse_list, "list"
                | parse_record, "record"
                | parse_par_expr, "expression"
                | parse_func_ref, "function reference"
                | parse_lambda, "lambda"
        }?;

        token!("`@`"; self: Token::Symbol { value: '@', .. })?;

        let index = cut!(self: parse_expr)?;

        let range = (expr.range().0, index.range().1);

        Ok(AstNode::IndexingOp {
            expr: Box::new(expr),
            index: Box::new(index),
            range,
        })
    }

    fn parse_concat(&mut self) -> PResult {
        let left = any! {
            self: parse_range, "range"
                | parse_indexing, "indexing"
                | parse_identifier, "identifier"
                | parse_bool, "boolean"
                | parse_string, "string"
                | parse_number, "number"
                | parse_list, "list"
                | parse_record, "record"
                | parse_par_expr, "expression"
                | parse_func_ref, "function reference"
                | parse_lambda, "lambda"
        }?;

        token!("`++`"; self: Token::ConcatOperator { .. })?;

        let right = cut!(self: parse_expr)?;

        let range = (left.range().0, right.range().1);

        Ok(AstNode::ConcatOp {
            left: Box::new(left),
            right: Box::new(right),
            range,
        })
    }

    fn parse_iteration_op(&mut self) -> PResult {
        let left = self.parse_expr()?;

        token!("`|`"; self: Token::Symbol { value: '|', .. })?;

        let id = cut!(self: parse_identifier)?;

        let delim = token!("`|`"; self: Token::Symbol { value: '|', .. });
        self.cut(delim)?;

        let right = any! {
            self: parse_block, "indented block"
                | parse_case, "case statement"
                | parse_func_call, "function call"
                | parse_expr_stmt, "expression"
        };
        let right = self.cut(right)?;

        let range = (left.range().0, right.range().1);

        Ok(AstNode::IterationOp {
            id: Box::new(id),
            expr: Box::new(left),
            body: Box::new(right),
            range,
        })
    }

    fn parse_identifier(&mut self) -> PResult {
        map! {
            self: Token::Identifier { name, span }, "identifier" => {
                AstNode::Ident {
                    name,
                    range: span.range,
                }
            }
        }
    }

    fn parse_bool(&mut self) -> PResult {
        let value = token!("boolean"; self: Token::Keyword { name: Keyword::True, .. }
                                          | Token::Keyword { name: Keyword::False, .. })?;

        let (value, span) = match value {
            Token::Keyword { name, span } => match name {
                Keyword::True => (true, span),
                Keyword::False => (false, span),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        Ok(AstNode::BoolLit {
            value,
            range: span.range,
        })
    }

    fn parse_string(&mut self) -> PResult {
        let format_marker = opt!(self: Token::Symbol { value: '$', .. });

        let (value, span) = self.parse_string_part()?;

        Ok(AstNode::StrLit {
            value,
            is_format: format_marker.is_some(),
            range: span.range,
        })
    }

    fn parse_string_part(&mut self) -> Result<(EcoString, Span), ParseError> {
        map!(self: Token::String { value, span }, "string" => (value, span))
    }

    fn parse_regex(&mut self) -> PResult {
        map!(self: Token::Regex {
            value,
            is_case_insensitive,
            is_global,
            is_multiline,
            is_dotall,
            is_sticky,
            span,
        }, "regular expression" => {
            AstNode::Regex {
                value,
                is_case_insensitive,
                is_global,
                is_multiline,
                is_dotall,
                is_sticky,
                range: span.range,
            }
        })
    }

    fn parse_number(&mut self) -> PResult {
        let (whole, whole_span) = self.parse_number_part()?;

        if opt!(self: Token::Symbol { value: '.', .. }).is_none() {
            return Ok(AstNode::NumLit {
                value: whole as f64,
                range: whole_span.range,
            });
        }

        let (fraction, fraction_span) = cut!(self: parse_number_part)?;

        Ok(AstNode::NumLit {
            value: format!("{}.{}", whole, fraction).parse::<f64>().unwrap(),
            range: (whole_span.range.0, fraction_span.range.1),
        })
    }

    fn parse_number_part(&mut self) -> Result<(usize, Span), ParseError> {
        map!(self: Token::Number { value, span }, "number" => (value, span))
    }

    fn parse_list(&mut self) -> PResult {
        let start = token!("`[`"; self: Token::Symbol { value: '[', .. })?;

        self.ignore_nl_and_ws()?;

        let items = cut!(self: parse_list_items)?;

        let end = self.parse_list_end()?;

        Ok(AstNode::List {
            items: items.into(),
            range: (start.span().range.0, end.span().range.1),
        })
    }

    fn parse_list_end(&mut self) -> Result<Token, ParseError> {
        self.ignore_nl_and_ws()?;

        token!("`]`"; self: Token::Symbol { value: ']', .. })
    }

    fn parse_list_items(&mut self) -> Result<Vec<AstNode>, ParseError> {
        let mut items = Vec::new();

        if peek!(0, self: Token::Symbol { value: ']', .. }).is_some() {
            return Ok(items);
        }

        loop {
            items.push(cut!(self: parse_expr)?);

            if opt!(self: parse_list_record_sep).is_some() {
                if peek!(self: parse_list_end).is_ok() {
                    break;
                }
            } else {
                if peek!(self: parse_list_end).is_ok() {
                    break;
                }

                return Err(ParseError::Unexpected {
                    expected: &["`,`", "`]`"],
                });
            }
        }

        Ok(items)
    }

    fn parse_record(&mut self) -> PResult {
        let start = token!("`{`"; self: Token::Symbol { value: '{', .. })?;

        self.ignore_nl_and_ws()?;

        let (keys, values) = cut!(self: parse_record_items)?;

        self.ignore_nl_and_ws()?;

        let end = self.parse_record_end()?;

        Ok(AstNode::Record {
            keys: keys.into(),
            values: values.into(),
            range: (start.span().range.0, end.span().range.1),
        })
    }

    fn parse_record_end(&mut self) -> Result<Token, ParseError> {
        self.ignore_nl_and_ws()?;

        token!("`}`"; self: Token::Symbol { value: '}', .. })
    }

    fn parse_record_items(&mut self) -> Result<(Vec<AstNode>, Vec<AstNode>), ParseError> {
        let mut keys = Vec::new();
        let mut values = Vec::new();

        if peek!(0, self: Token::Symbol { value: '}', .. }).is_some() {
            return Ok((keys, values));
        }

        loop {
            let key = cut!(self: parse_identifier)?;

            let eq = token!("`:`"; self: Token::Symbol { value: ':', .. });
            self.cut(eq)?;

            let value = cut!(self: parse_expr)?;

            keys.push(key);
            values.push(value);

            if opt!(self: parse_list_record_sep).is_some() {
                if peek!(self: parse_record_end).is_ok() {
                    break;
                }
            } else {
                if peek!(self: parse_record_end).is_ok() {
                    break;
                }

                return Err(ParseError::Unexpected {
                    expected: &["`,`", "`}`"],
                });
            }
        }

        Ok((keys, values))
    }

    fn parse_list_record_sep(&mut self) -> Result<(), ParseError> {
        self.ignore_nl_and_ws()?;

        let comma = token!("`,`"; self: Token::Symbol { value: ',', .. });
        self.cut(comma)?;

        self.ignore_nl_and_ws()?;

        Ok(())
    }

    fn parse_range(&mut self) -> PResult {
        let start = self.parse_range_part()?;

        let op = token!("range operator"; self: Token::RangeOperator { .. })?;
        let mode = match op {
            Token::RangeOperator { mode, .. } => mode,
            _ => unreachable!(),
        };

        let end = cut!(self: parse_range_part)?;

        let range = (start.range().0, end.range().1);

        Ok(AstNode::Range {
            mode,
            start: Box::new(start),
            end: Box::new(end),
            range,
        })
    }

    fn parse_range_part(&mut self) -> PResult {
        any! {
            self: parse_identifier, "identifier"
                | parse_string, "string"
                | parse_number, "number"
                | parse_par_expr, "expression"
        }
    }

    fn parse_par_expr(&mut self) -> PResult {
        token!("`(`"; self: Token::Symbol { value: '(', .. })?;

        self.ignore_nl_and_ws()?;

        let expr = any! {
            self: parse_indexing, "indexing"
                | parse_concat, "concatenation"
                | parse_iteration_op, "iterator"
                | parse_func_call, "function call"
                | parse_expr, "expression"
        };
        let expr = self.cut(expr)?;

        self.ignore_nl_and_ws()?;

        let end = token!("`)`"; self: Token::Symbol { value: ')', .. });
        self.cut(end)?;

        Ok(expr)
    }

    fn parse_func_ref(&mut self) -> PResult {
        let start = token!("`&`"; self: Token::Symbol { value: '&', .. })?;

        let id = cut!(self: parse_identifier)?;

        let range = (start.span().range.0, id.range().1);

        Ok(AstNode::FuncRef {
            id: Box::new(id),
            range,
        })
    }
}
