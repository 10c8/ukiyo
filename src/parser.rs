use std::fmt::Write;

use crate::lexer::*;

type Range = (usize, usize);

#[derive(Debug, Clone, PartialEq)]
pub enum CasePatternKind {
    Empty,
    Expr,
    Or,
}

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
    Block {
        body: Vec<AstNode<'a>>,
        range: Range,
    },
    FuncDecl {
        id: Option<Box<AstNode<'a>>>,
        params: Vec<AstNode<'a>>,
        body: Box<AstNode<'a>>,
        range: Range,
    },
    Case {
        expr: Box<AstNode<'a>>,
        cases: Vec<AstNode<'a>>,
        range: Range,
    },
    CaseBranch {
        pattern: Box<AstNode<'a>>,
        body: Box<AstNode<'a>>,
        range: Range,
    },
    CasePattern {
        kind: CasePatternKind,
        expr: Box<AstNode<'a>>,
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
    List {
        items: Vec<AstNode<'a>>,
        range: Range,
    },
    Record {
        keys: Vec<AstNode<'a>>,
        values: Vec<AstNode<'a>>,
        range: Range,
    },
}

impl<'a> AstNode<'a> {
    pub fn range(&self) -> Range {
        match self {
            AstNode::Prog { range, .. }
            | AstNode::ExprStmt { range, .. }
            | AstNode::Block { range, .. }
            | AstNode::FuncDecl { range, .. }
            | AstNode::Case { range, .. }
            | AstNode::CaseBranch { range, .. }
            | AstNode::CasePattern { range, .. }
            | AstNode::FuncCall { range, .. }
            | AstNode::Ident { range, .. }
            | AstNode::StrLit { range, .. }
            | AstNode::NumLit { range, .. }
            | AstNode::List { range, .. }
            | AstNode::Record { range, .. } => *range,
        }
    }
}

type PResult = Result<AstNode<'static>, PError>;

#[derive(Debug, Clone)]
pub enum PError {
    ParseError,
    SyntaxError { expected: &'static [&'static str] },
    InvalidIndent(usize),
    InvalidDedent(usize),

    Cut(Box<PError>, Token),
}

/// Runs a parser, propagating unrecoverable errors. (For internal use by macros.)
macro_rules! parse_cut {
    ($self:ident: $parser:ident) => {{
        let cursor = $self.lexer.cursor();
        match $self.$parser() {
            Ok(success) => Ok(success),
            Err(err) => {
                if let PError::Cut(_, _) = err {
                    // Unrecoverable error, propagate it
                    $self.set_cut_error(Some(err));
                    return Err(PError::ParseError);
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
            Err(PError::Cut(inner, span)) => Err(PError::Cut(inner, span)),
            Err(err) => Err(PError::Cut(Box::new(err), $self.lexer.peek())),
        }
    };
}

/// Matches the next token against a pattern. On success, returns `Token`.
macro_rules! token {
    ($name:literal; $self:ident: $token:pat) => {
        match $self.lexer.peek() {
            $token => Ok($self.lexer.next()),
            _ => {
                return Err(PError::SyntaxError { expected: &[$name] });
            }
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

        if result.is_err() {
            Err(result.err().unwrap())
        } else {
            Ok(result.ok().unwrap())
        }
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
    ($self:ident: $first:ident, $first_exp:literal $(| $rest:ident, $rest_exp:literal )*) => {{
        if let Ok(success) = parse_cut!($self: $first) {
            Ok(success)
        } $(else if let Ok(success) = parse_cut!($self: $rest) {
            Ok(success)
        })* else {
            Err(PError::SyntaxError {
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
            let result = parse_cut!($self: $parser);
            if result.is_err() {
                break;
            }

            results.push(result.unwrap());

            if opt!($self: $sep).is_none() {
                break;
            }
        }

        if !$n.contains(&results.len()) {
            return Err(PError::ParseError);
        }

        Ok(results)
    }};
    ($n:expr, $self:ident: $parser:ident / $sep:pat) => {{
        let mut results = Vec::new();

        loop {
            let result = parse_cut!($self: $parser);
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
            return Err(PError::ParseError);
        }

        Ok(results)
    }};
}

/// Matches a parser or token `n` times. On success, returns `Vec<T>`.
macro_rules! many {
    ($n:expr, $self:ident: $parser:ident) => {{
        let mut results = Vec::new();

        loop {
            let result = parse_cut!($self: $parser).map(|success| results.push(success));
            if result.is_err() {
                break;
            }
        }

        if !$n.contains(&results.len()) {
            return Err(PError::ParseError);
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
            return Err(PError::ParseError);
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
            _ => Err(PError::SyntaxError {
                expected: &[$($name),*],
            }),
        }
    };
}

pub struct Parser {
    lexer: Lexer,
    cut_error: Option<PError>,
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

        if self.lexer.peek() != Token::EOF {
            return Err(PError::ParseError);
        }

        Ok(program)
    }

    pub fn display_error(&mut self, error: PError) -> Result<String, std::fmt::Error> {
        let mut msg = String::new();

        let tok;
        let span;
        let err;

        if let Some(PError::Cut(inner_err, inner_tok)) = self.cut_error.take() {
            tok = inner_tok;
            span = inner_tok.span();
            err = *inner_err;
        } else {
            tok = self.lexer.peek();
            span = tok.span();
            err = error;
        }

        let line = span.line;
        let column = span.column;
        let start = span.range.0;
        let end = span.range.1;

        match err {
            PError::ParseError => {
                write!(&mut msg, "Parsing Error @ {}:{}\n", line, column)?;
                self.display_error_line(&mut msg, line, column, start, end);

                write!(&mut msg, "Unexpected {}.", tok.to_string())?;
            }
            PError::SyntaxError { expected } => {
                write!(&mut msg, "Syntax Error @ {}:{}\n", line, column)?;
                self.display_error_line(&mut msg, line, column, start, end);

                write!(&mut msg, "Unexpected {}. Expected one of:", tok)?;
                for e in expected {
                    write!(&mut msg, "\n - {}", e)?;
                }
            }
            PError::InvalidIndent(size) => {
                write!(&mut msg, "Indentation Error @ {}:{}\n", line, column)?;
                self.display_error_line(&mut msg, line, column, start, end);

                write!(
                    &mut msg,
                    "Expected indent of size {}, found {}.",
                    self.indent_stack.last().unwrap_or(&0) + self.indent_size,
                    size
                )?;
            }
            PError::InvalidDedent(size) => {
                write!(&mut msg, "Indentation Error @ {}:{}\n", line, column)?;
                self.display_error_line(&mut msg, line, column, start, end);

                write!(
                    &mut msg,
                    "Expected indent of size {}, found {}.",
                    self.indent_stack.last().unwrap_or(&0),
                    size
                )?;
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

    fn set_cut_error(&mut self, err: Option<PError>) {
        if self.cut_error.is_none() {
            self.cut_error = err;
        }
    }

    /// Transforms a recoverable error into an unrecoverable one.
    fn cut<T>(&mut self, result: Result<T, PError>) -> Result<T, PError> {
        match result {
            Ok(success) => Ok(success),
            Err(PError::Cut(inner, span)) => Err(PError::Cut(inner, span)),
            Err(err) => Err(PError::Cut(Box::new(err), self.lexer.peek())),
        }
    }

    fn ignore_nl(&mut self) -> Result<(), PError> {
        many!(0.., self: Token::Newline { .. })?;
        Ok(())
    }

    fn ignore_nl_and_ws(&mut self) -> Result<(), PError> {
        many!(0.., self: Token::Newline { .. } | Token::Indent { .. } | Token::Dedent { .. })?;
        Ok(())
    }

    fn parse_indent(&mut self) -> Result<(), PError> {
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
                self.cut(Err(PError::InvalidIndent(size)))
            }
        } else {
            Err(PError::SyntaxError {
                expected: &["indent"],
            })
        }
    }

    fn parse_dedent(&mut self) -> Result<(), PError> {
        let last_indent = self.indent_stack.last().unwrap_or(&0);

        let dedent = token!("dedent"; self: Token::Dedent { .. })?;
        if let Token::Dedent { size, .. } = dedent {
            if size == *last_indent {
                self.indent_stack.pop();
                Ok(())
            } else {
                self.cut(Err(PError::InvalidDedent(size)))
            }
        } else {
            Err(PError::SyntaxError {
                expected: &["dedent"],
            })
        }
    }

    fn parse_program(&mut self) -> PResult {
        self.ignore_nl()?;

        let body = sep!(0.., self: parse_stmt / Token::Newline { .. })?;

        self.ignore_nl()?;

        let range = (0, body.last().map(|stmt| stmt.range().1).unwrap_or(0));

        Ok(AstNode::Prog { body, range })
    }

    fn parse_stmt(&mut self) -> PResult {
        any! {
            self: parse_func_decl, "function declaration"
                | parse_case, "case"
                | parse_expr_stmt, "expression"
        }
    }

    fn parse_expr_stmt(&mut self) -> PResult {
        let expr = any! {
            self: parse_func_call, "function call"
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

        let mut body = Vec::new();

        loop {
            body.push(cut!(self: parse_stmt)?);

            self.ignore_nl()?;

            if peek!(0, self: Token::Dedent { .. }).is_some() {
                break;
            }
        }

        cut!(self: parse_dedent)?;

        if body.is_empty() {
            return Err(PError::ParseError);
        }

        let range = (
            body.first().map(|stmt| stmt.range().0).unwrap_or(0),
            body.last().map(|stmt| stmt.range().1).unwrap_or(0),
        );

        Ok(AstNode::Block { body, range })
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

        map!(self: Token::AssignmentArrow { .. }, "->" => { })?;

        let body = any! {
            self: parse_block, "block"
                | parse_case, "case"
                | parse_expr, "expression"
        };
        let body = self.cut(body)?;

        let range = (range_start, body.range().1);

        Ok(AstNode::FuncDecl {
            id,
            params,
            body: Box::new(body),
            range,
        })
    }

    fn parse_case(&mut self) -> PResult {
        let case = token!("\"case\""; self: Token::Keyword { name: Keyword::Case, .. })?;

        let expr = cut!(self: parse_expr)?;

        let of = token!("\"of\""; self: Token::Keyword { name: Keyword::Of, .. });
        self.cut(of)?;

        cut!(self: parse_indent)?;

        let cases = cut!(self: parse_case_cases)?;

        cut!(self: parse_dedent)?;

        let range = (case.span().range.0, cases.last().unwrap().range().1);

        Ok(AstNode::Case {
            expr: Box::new(expr),
            cases,
            range,
        })
    }

    fn parse_case_cases(&mut self) -> Result<Vec<AstNode<'static>>, PError> {
        let mut cases = Vec::new();

        loop {
            cases.push(cut!(self: parse_case_branch)?);

            if peek!(0, self: Token::Dedent { .. }).is_some() {
                break;
            }

            if opt!(self: Token::Newline { .. }).is_none() {
                return Err(PError::SyntaxError {
                    expected: &["\"newline\"", "\"dedent\""],
                });
            }
        }

        if cases.is_empty() {
            return Err(PError::ParseError);
        }

        Ok(cases)
    }

    fn parse_case_branch(&mut self) -> PResult {
        let pattern = cut!(self: parse_case_pattern)?;

        let arrow = token!("\"=>\""; self: Token::MatchArrow { .. });
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
        if opt!(self: Token::Identifier { name: "_", .. }).is_some() {
            let range = self.lexer.peek().span().range;

            return Ok(AstNode::CasePattern {
                kind: CasePatternKind::Empty,
                expr: Box::new(AstNode::Ident { name: "_", range }),
                range,
            });
        }

        let first = cut!(self: parse_case_pattern_expr)?;

        if opt!(self: Token::Symbol { value: '|', .. }).is_some() {
            let rest = cut!(self: parse_case_pattern_or)?;

            let range = (first.range().0, rest.last().unwrap().range().1);

            let mut items = Vec::new();
            items.push(first);
            items.append(&mut rest.clone());

            Ok(AstNode::CasePattern {
                kind: CasePatternKind::Or,
                expr: Box::new(AstNode::List { items, range }),
                range,
            })
        } else {
            let range = first.range();

            Ok(AstNode::CasePattern {
                kind: CasePatternKind::Expr,
                expr: Box::new(first),
                range,
            })
        }
    }

    fn parse_case_body(&mut self) -> PResult {
        any! {
            self: parse_block, "block"
                | parse_func_call, "function call"
                | parse_expr, "expression"
        }
    }

    fn parse_case_pattern_expr(&mut self) -> PResult {
        any! {
            self: parse_identifier, "identifier"
                | parse_string, "string"
                | parse_number, "number"
        }
    }

    fn parse_case_pattern_or(&mut self) -> Result<Vec<AstNode<'static>>, PError> {
        let mut items = Vec::new();

        loop {
            let item = any! {
                self: parse_identifier, "identifier"
                    | parse_string, "string"
                    | parse_number, "number"
            };

            items.push(self.cut(item)?);

            if opt!(self: Token::Symbol { value: '|', .. }).is_none() {
                break;
            }
        }

        if items.is_empty() {
            return Err(PError::ParseError);
        }

        Ok(items)
    }

    fn parse_expr(&mut self) -> PResult {
        any! {
            self: parse_identifier, "identifier"
                | parse_string, "string"
                | parse_number, "number"
                | parse_list, "list"
                | parse_record, "record"
                | parse_par_expr, "expression"
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
        map! {
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
        map!(self: Token::String { value, span }, "string" => (value, span))
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

    fn parse_number_part(&mut self) -> Result<(u64, Span), PError> {
        map!(self: Token::Number { value, span }, "number" => (value, span))
    }

    fn parse_list(&mut self) -> PResult {
        let start = token!("\"[\""; self: Token::Symbol { value: '[', .. })?;

        self.ignore_nl_and_ws()?;

        let items = cut!(self: parse_list_items)?;

        let end = self.parse_list_end()?;

        Ok(AstNode::List {
            items,
            range: (start.span().range.0, end.span().range.1),
        })
    }

    fn parse_list_end(&mut self) -> Result<Token, PError> {
        self.ignore_nl_and_ws()?;

        token!("\"]\""; self: Token::Symbol { value: ']', .. })
    }

    fn parse_list_items(&mut self) -> Result<Vec<AstNode<'static>>, PError> {
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

                return Err(PError::SyntaxError {
                    expected: &["\",\"", "\"]\""],
                });
            }
        }

        Ok(items)
    }

    fn parse_record(&mut self) -> PResult {
        let start = token!("\"{\""; self: Token::Symbol { value: '{', .. })?;

        self.ignore_nl_and_ws()?;

        let (keys, values) = cut!(self: parse_record_items)?;

        self.ignore_nl_and_ws()?;

        let end = self.parse_record_end()?;

        Ok(AstNode::Record {
            keys,
            values,
            range: (start.span().range.0, end.span().range.1),
        })
    }

    fn parse_record_end(&mut self) -> Result<Token, PError> {
        self.ignore_nl_and_ws()?;

        token!("\"}\""; self: Token::Symbol { value: '}', .. })
    }

    fn parse_record_items(
        &mut self,
    ) -> Result<(Vec<AstNode<'static>>, Vec<AstNode<'static>>), PError> {
        let mut keys = Vec::new();
        let mut values = Vec::new();

        if peek!(0, self: Token::Symbol { value: '}', .. }).is_some() {
            return Ok((keys, values));
        }

        loop {
            let key = cut!(self: parse_string)?;

            let eq = token!("\"=\""; self: Token::Symbol { value: '=', .. });
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

                return Err(PError::SyntaxError {
                    expected: &["\",\"", "\"}\""],
                });
            }
        }

        Ok((keys, values))
    }

    fn parse_list_record_sep(&mut self) -> Result<(), PError> {
        self.ignore_nl_and_ws()?;

        let comma = token!("\",\""; self: Token::Symbol { value: ',', .. });
        self.cut(comma)?;

        self.ignore_nl_and_ws()?;

        Ok(())
    }

    fn parse_par_expr(&mut self) -> PResult {
        let start = token!("\"(\""; self: Token::Symbol { value: '(', .. })?;

        self.ignore_nl_and_ws()?;

        let expr = any! {
            self: parse_func_call, "function call"
                | parse_expr, "expression"
        };
        let expr = self.cut(expr)?;

        self.ignore_nl_and_ws()?;

        let end = token!("\")\""; self: Token::Symbol { value: ')', .. })?;

        let expr = match expr {
            // TODO: Add `FuncDecl` for anonymous functions
            // TODO: Add `Case`
            AstNode::FuncCall { callee, arg, .. } => AstNode::FuncCall {
                callee,
                arg,
                range: (start.span().range.0, end.span().range.1),
            },
            AstNode::Ident { name, .. } => AstNode::Ident {
                name,
                range: (start.span().range.0, end.span().range.1),
            },
            AstNode::StrLit { value, format, .. } => AstNode::StrLit {
                value,
                format,
                range: (start.span().range.0, end.span().range.1),
            },
            AstNode::NumLit { value, .. } => AstNode::NumLit {
                value,
                range: (start.span().range.0, end.span().range.1),
            },
            AstNode::List { items, .. } => AstNode::List {
                items,
                range: (start.span().range.0, end.span().range.1),
            },
            AstNode::Record { keys, values, .. } => AstNode::Record {
                keys,
                values,
                range: (start.span().range.0, end.span().range.1),
            },
            _ => expr,
        };

        Ok(expr)
    }
}
