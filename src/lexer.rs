use crate::scanner::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub range: (usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Case,
    Do,
    Each,
    Of,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOperator {
    Inclusive,
    Exclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Identifier {
        // [a-zA-Z_][a-zA-Z0-9_]*
        name: &'static str,
        span: Span,
    },
    Keyword {
        // case | do | each | of
        name: Keyword,
        span: Span,
    },
    String {
        // "[^"]*"
        value: &'static str,
        span: Span,
    },
    Regex {
        // /[^/]+/
        value: &'static str,
        span: Span,
    },
    Number {
        // [0-9]+
        value: u64,
        span: Span,
    },
    Symbol {
        // ',' | '.' | '=' | '$' | ':' | '\' | '|' | '(' | ')' | '[' | ']' | '{' | '}'
        value: char,
        span: Span,
    },
    RangeOperator {
        // "..=" | "..<"
        mode: RangeOperator,
        span: Span,
    },
    AssignmentArrow {
        // ->
        span: Span,
    },
    ApplicationArrow {
        // <|
        span: Span,
    },
    MatchArrow {
        // =>
        span: Span,
    },
    // Context (%%) is treated as an identifier
    Newline {
        span: Span,
    },
    Indent {
        size: usize,
        span: Span,
    },
    Dedent {
        size: usize,
        span: Span,
    },
    EOF {
        span: Span,
    },
}

impl Token {
    pub fn span(&self) -> Span {
        match self {
            Token::Identifier { span, .. }
            | Token::Keyword { span, .. }
            | Token::String { span, .. }
            | Token::Regex { span, .. }
            | Token::Number { span, .. }
            | Token::Symbol { span, .. }
            | Token::RangeOperator { span, .. }
            | Token::AssignmentArrow { span, .. }
            | Token::ApplicationArrow { span, .. }
            | Token::MatchArrow { span, .. }
            | Token::Newline { span, .. }
            | Token::Indent { span, .. }
            | Token::Dedent { span, .. }
            | Token::EOF { span } => *span,
        }
    }
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Token::Identifier { .. } => write!(f, "identifier"),
            Token::Keyword { .. } => write!(f, "keyword"),
            Token::String { .. } => write!(f, "string"),
            Token::Regex { .. } => write!(f, "regex"),
            Token::Number { .. } => write!(f, "number"),
            Token::Symbol { value, .. } => write!(f, "`{}`", value),
            Token::RangeOperator { .. } => write!(f, "range"),
            Token::AssignmentArrow { .. } => write!(f, "`->`"),
            Token::ApplicationArrow { .. } => write!(f, "`<|`"),
            Token::MatchArrow { .. } => write!(f, "`=>`"),
            Token::Newline { .. } => write!(f, "newline"),
            Token::Indent { .. } => write!(f, "indent"),
            Token::Dedent { .. } => write!(f, "dedent"),
            Token::EOF { .. } => write!(f, "end of file"),
        }
    }
}

#[derive(Debug)]
pub enum LexError {
    UnexpectedChar(char, usize, usize),
    UnexpectedEOF(usize, usize),
    UnexpectedIndent(usize, usize, usize),
    NumberOverflow(usize, usize),
}

#[derive(Clone)]
pub struct Lexer {
    scanner: Scanner,
    tokens: Vec<Token>,
    cursor: usize,
    source: String,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let scanner = Scanner::new(input);
        let tokens = Vec::new();

        Self {
            scanner,
            tokens,
            cursor: 0,
            source: input.to_string(),
        }
    }

    pub fn lex(&mut self) -> Result<(), LexError> {
        let mut tokens = Vec::new();

        let mut indent_stack = Vec::new();
        let mut tab_size = 0;

        while let Some(chr) = self.scanner.peek() {
            let span_start = self.scanner.cursor();
            let line = self.scanner.line();
            let column = self.scanner.column();

            match chr {
                // Comment
                '#' => {
                    self.scanner.consume_while(|chr| *chr != '\n');
                    self.scanner.consume_while(|chr| *chr == '\n');
                }
                // Newline (indentation)
                '\n' => {
                    self.scanner.consume_while(|chr| *chr == '\n');

                    let mut spaces = 0;

                    while let Some(chr) = self.scanner.peek() {
                        if *chr == ' ' {
                            spaces += 1;
                            self.scanner.next();
                        } else {
                            break;
                        }
                    }

                    let span = Span {
                        line,
                        column,
                        range: (span_start, self.scanner.cursor()),
                    };

                    if tab_size == 0 {
                        tab_size = spaces;
                    }

                    if tab_size == 0 {
                        tokens.push(Token::Newline { span });
                        continue;
                    }

                    if spaces == 0 && !indent_stack.is_empty() {
                        while let Some(last_indent) = indent_stack.last() {
                            tokens.push(Token::Dedent {
                                size: *last_indent,
                                span,
                            });
                            indent_stack.pop();
                        }

                        tokens.push(Token::Newline { span });
                        continue;
                    }

                    let last_indent = *indent_stack.last().unwrap_or(&0);

                    if spaces > last_indent {
                        indent_stack.push(spaces);
                        tokens.push(Token::Indent { size: spaces, span });
                    } else if spaces < last_indent {
                        while let Some(last_indent) = indent_stack.last() {
                            if spaces < *last_indent {
                                tokens.push(Token::Dedent {
                                    size: *last_indent,
                                    span,
                                });
                                indent_stack.pop();
                            } else {
                                tokens.push(Token::Newline { span });
                                break;
                            }
                        }
                    } else {
                        tokens.push(Token::Newline { span });
                    }
                }
                // Whitespace
                chr if chr.is_whitespace() => {
                    self.scanner.consume_while(|chr| chr.is_whitespace());
                }
                // Identifier
                chr if chr.is_ascii_alphabetic() || *chr == '_' => {
                    let mut identifier = String::new();

                    identifier.push(*chr);
                    self.scanner.next();

                    while let Some(chr) = self.scanner.peek() {
                        if chr.is_ascii_alphanumeric() || *chr == '_' {
                            identifier.push(*chr);
                            self.scanner.next();
                        } else {
                            break;
                        }
                    }

                    let span = Span {
                        line,
                        column,
                        range: (span_start, self.scanner.cursor()),
                    };

                    let token = match identifier.as_str() {
                        "case" => Token::Keyword {
                            name: Keyword::Case,
                            span,
                        },
                        "do" => Token::Keyword {
                            name: Keyword::Do,
                            span,
                        },
                        "each" => Token::Keyword {
                            name: Keyword::Each,
                            span,
                        },
                        "of" => Token::Keyword {
                            name: Keyword::Of,
                            span,
                        },
                        _ => Token::Identifier {
                            name: Box::leak(identifier.into_boxed_str()),
                            span,
                        },
                    };
                    tokens.push(token);
                }
                // Number
                num if num.is_digit(10) => {
                    let mut number = String::new();

                    number.push(*num);
                    self.scanner.next();

                    while let Some(chr) = self.scanner.peek() {
                        if chr.is_digit(10) {
                            number.push(*chr);
                            self.scanner.next();
                        } else {
                            break;
                        }
                    }

                    let span = Span {
                        line,
                        column,
                        range: (span_start, self.scanner.cursor()),
                    };

                    let number = number.parse::<u64>().map_err(|_| {
                        LexError::NumberOverflow(self.scanner.column(), self.scanner.line())
                    })?;
                    tokens.push(Token::Number {
                        value: number,
                        span,
                    });
                }
                // String
                '"' => {
                    let mut string = String::new();

                    self.scanner.next();

                    while let Some(chr) = self.scanner.peek() {
                        match chr {
                            '\\' => {
                                self.scanner.next();

                                if let Some(chr) = self.scanner.peek() {
                                    match chr {
                                        '\\' | '"' => string.push(*chr),
                                        'n' => string.push('\n'),
                                        't' => string.push('\t'),
                                        _ => {
                                            return Err(LexError::UnexpectedChar(
                                                *chr,
                                                self.scanner.column(),
                                                self.scanner.line(),
                                            ))
                                        }
                                    }

                                    self.scanner.next();
                                } else {
                                    return Err(LexError::UnexpectedEOF(
                                        self.scanner.column(),
                                        self.scanner.line(),
                                    ));
                                }
                            }
                            '"' => {
                                self.scanner.next();
                                break;
                            }
                            _ => {
                                string.push(*chr);
                                self.scanner.next();
                            }
                        }
                    }

                    let span = Span {
                        line,
                        column,
                        range: (span_start, self.scanner.cursor()),
                    };

                    tokens.push(Token::String {
                        value: Box::leak(string.into_boxed_str()),
                        span,
                    });
                }
                // Multiline String
                '@' => {
                    self.scanner.next();

                    if self.scanner.try_consume_sequence("\"\n") {
                        let mut string = String::new();

                        let last_indent = *indent_stack.last().unwrap_or(&0);
                        let mut discarded_indent = 0;

                        while discarded_indent < last_indent {
                            if self.scanner.try_consume(' ') {
                                discarded_indent += 1;
                            } else {
                                break;
                            }
                        }

                        while let Some(_) = self.scanner.peek() {
                            if self.scanner.try_consume('\n') {
                                let mut spaces = 0;

                                if last_indent > 0 {
                                    while let Some(' ') = self.scanner.peek_nth(spaces) {
                                        spaces += 1;
                                    }

                                    if spaces < last_indent {
                                        return Err(LexError::UnexpectedIndent(
                                            spaces,
                                            self.scanner.line(),
                                            self.scanner.column(),
                                        ));
                                    } else {
                                        discarded_indent = 0;

                                        while discarded_indent < last_indent {
                                            if self.scanner.try_consume(' ') {
                                                discarded_indent += 1;
                                            } else {
                                                break;
                                            }
                                        }
                                    }
                                }

                                if self.scanner.try_consume_sequence("\"@") {
                                    if tab_size > 0 {
                                        if spaces != last_indent {
                                            return Err(LexError::UnexpectedIndent(
                                                spaces,
                                                self.scanner.line(),
                                                self.scanner.column(),
                                            ));
                                        }
                                    }

                                    break;
                                } else {
                                    string.push('\n');
                                }
                            } else {
                                if self.scanner.try_consume_sequence("\"@") {
                                    return Err(LexError::UnexpectedChar(
                                        '"',
                                        self.scanner.column(),
                                        self.scanner.line(),
                                    ));
                                }

                                string.push(*self.scanner.next().unwrap());
                            }
                        }

                        let span = Span {
                            line,
                            column,
                            range: (span_start, self.scanner.cursor()),
                        };

                        tokens.push(Token::String {
                            value: Box::leak(string.into_boxed_str()),
                            span,
                        });
                    } else {
                        return Err(LexError::UnexpectedChar(
                            *self.scanner.peek().unwrap(),
                            self.scanner.column(),
                            self.scanner.line(),
                        ));
                    }
                }
                // Regex
                '/' => {
                    let mut regex = String::new();

                    self.scanner.next();

                    while let Some(chr) = self.scanner.peek() {
                        match chr {
                            '\\' => {
                                self.scanner.next();

                                if let Some(chr) = self.scanner.peek() {
                                    match chr {
                                        '\\' | '/' => regex.push(*chr),
                                        'n' => regex.push('\n'),
                                        't' => regex.push('\t'),
                                        _ => {
                                            return Err(LexError::UnexpectedChar(
                                                *chr,
                                                self.scanner.column(),
                                                self.scanner.line(),
                                            ))
                                        }
                                    }

                                    self.scanner.next();
                                } else {
                                    return Err(LexError::UnexpectedEOF(
                                        self.scanner.column(),
                                        self.scanner.line(),
                                    ));
                                }
                            }
                            '/' => {
                                self.scanner.next();
                                break;
                            }
                            _ => {
                                regex.push(*chr);
                                self.scanner.next();
                            }
                        }
                    }

                    let span = Span {
                        line,
                        column,
                        range: (span_start, self.scanner.cursor()),
                    };

                    tokens.push(Token::Regex {
                        value: Box::leak(regex.into_boxed_str()),
                        span,
                    });
                }
                // Assignment Arrow
                '-' => {
                    self.scanner.next();

                    if self.scanner.try_consume('>') {
                        tokens.push(Token::AssignmentArrow {
                            span: Span {
                                line,
                                column,
                                range: (span_start, self.scanner.cursor()),
                            },
                        });
                    } else {
                        return Err(LexError::UnexpectedChar(
                            *self.scanner.peek().unwrap(),
                            self.scanner.column(),
                            self.scanner.line(),
                        ));
                    }
                }
                // Symbol / Match Arrow
                '=' => {
                    self.scanner.next();

                    if self.scanner.try_consume('>') {
                        tokens.push(Token::MatchArrow {
                            span: Span {
                                line,
                                column,
                                range: (span_start, self.scanner.cursor()),
                            },
                        });
                    } else {
                        tokens.push(Token::Symbol {
                            value: '=',
                            span: Span {
                                line,
                                column,
                                range: (span_start, self.scanner.cursor()),
                            },
                        });
                    }
                }
                // Symbol
                ',' | ':' | '$' | '\\' | '|' | '(' | ')' | '[' | ']' | '{' | '}' => {
                    tokens.push(Token::Symbol {
                        value: *self.scanner.next().unwrap(),
                        span: Span {
                            line,
                            column,
                            range: (span_start, self.scanner.cursor()),
                        },
                    });
                }
                // Symbol / Range Operator
                '.' => {
                    self.scanner.next();

                    if self.scanner.try_consume('.') {
                        if self.scanner.try_consume('=') {
                            tokens.push(Token::RangeOperator {
                                mode: RangeOperator::Inclusive,
                                span: Span {
                                    line,
                                    column,
                                    range: (span_start, self.scanner.cursor()),
                                },
                            });
                        } else if self.scanner.try_consume('<') {
                            tokens.push(Token::RangeOperator {
                                mode: RangeOperator::Exclusive,
                                span: Span {
                                    line,
                                    column,
                                    range: (span_start, self.scanner.cursor()),
                                },
                            });
                        } else {
                            if let Some(invalid) = self.scanner.peek() {
                                return Err(LexError::UnexpectedChar(
                                    *invalid,
                                    self.scanner.column(),
                                    self.scanner.line(),
                                ));
                            } else {
                                return Err(LexError::UnexpectedEOF(
                                    self.scanner.column(),
                                    self.scanner.line(),
                                ));
                            }
                        }
                    } else {
                        tokens.push(Token::Symbol {
                            value: '.',
                            span: Span {
                                line,
                                column,
                                range: (span_start, self.scanner.cursor()),
                            },
                        });
                    }
                }
                // Expression Applicator
                '<' => {
                    self.scanner.next();

                    if self.scanner.try_consume('|') {
                        tokens.push(Token::ApplicationArrow {
                            span: Span {
                                line,
                                column,
                                range: (span_start, self.scanner.cursor()),
                            },
                        });
                    } else {
                        return Err(LexError::UnexpectedChar(
                            *self.scanner.peek().unwrap(),
                            self.scanner.column(),
                            self.scanner.line(),
                        ));
                    }
                }
                // Context
                '%' => {
                    self.scanner.next();

                    if self.scanner.try_consume('%') {
                        tokens.push(Token::Identifier {
                            name: "%%",
                            span: Span {
                                line,
                                column,
                                range: (span_start, self.scanner.cursor()),
                            },
                        });
                    } else {
                        return Err(LexError::UnexpectedChar(
                            *self.scanner.peek().unwrap(),
                            self.scanner.column(),
                            self.scanner.line(),
                        ));
                    }
                }
                // Unknown
                _ => {
                    return Err(LexError::UnexpectedChar(
                        *chr,
                        self.scanner.column(),
                        self.scanner.line(),
                    ));
                }
            };
        }

        self.tokens = tokens;

        Ok(())
    }

    /// Returns the current cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Sets the cursor position.
    pub fn set_cursor(&mut self, cursor: usize) {
        self.cursor = cursor;
    }

    /// Returns the source code.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the token at the current cursor position.
    pub fn peek(&mut self) -> Token {
        self.tokens.get(self.cursor).copied().unwrap_or(Token::EOF {
            span: Span {
                line: self.scanner.line(),
                column: self.scanner.column(),
                range: (self.scanner.cursor(), self.scanner.cursor()),
            },
        })
    }

    /// Returns the token at the given cursor position.
    pub fn peek_nth(&mut self, n: usize) -> Token {
        self.tokens
            .get(self.cursor + n)
            .copied()
            .unwrap_or(Token::EOF {
                span: Span {
                    line: self.scanner.line(),
                    column: self.scanner.column(),
                    range: (self.scanner.cursor(), self.scanner.cursor()),
                },
            })
    }

    /// Returns the token at the current cursor position and advances the cursor.
    pub fn next(&mut self) -> Token {
        let token = self.tokens.get(self.cursor).copied().unwrap_or(Token::EOF {
            span: Span {
                line: self.scanner.line(),
                column: self.scanner.column(),
                range: (self.scanner.cursor(), self.scanner.cursor()),
            },
        });
        self.cursor += 1;

        token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new(
            r#"# this is a comment
test -> "test" # this is an inline comment
# this is another comment"#,
        );
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Identifier {
                name: "test",
                span: Span {
                    line: 2,
                    column: 1,
                    range: (20, 24),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::AssignmentArrow {
                span: Span {
                    line: 2,
                    column: 6,
                    range: (25, 27),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::String {
                value: "test",
                span: Span {
                    line: 2,
                    column: 9,
                    range: (28, 34),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_identifier() {
        let mut lexer = Lexer::new("hello_World23");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Identifier {
                name: "hello_World23",
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 13),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_keyword() {
        let mut lexer = Lexer::new("case");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Keyword {
                name: Keyword::Case,
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 4),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_string() {
        let mut lexer = Lexer::new(r#""hello world\n""#);
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::String {
                value: "hello world\n",
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 15),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_escaped_string() {
        let mut lexer = Lexer::new(r#""hello \"world\"\n"#);
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::String {
                value: "hello \"world\"\n",
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 18),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_multiline_string() {
        let mut lexer = Lexer::new(
            r#"
  """
  hello "world"
  this is cool!
  """"#,
        );
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Indent {
                size: 2,
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 3),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::String {
                value: "hello \"world\"\nthis is cool!",
                span: Span {
                    line: 2,
                    column: 3,
                    range: (3, 44),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_regex() {
        let mut lexer = Lexer::new(r#"/[a-zA-Z]+/"#);
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Regex {
                value: "[a-zA-Z]+",
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 11),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_number() {
        let mut lexer = Lexer::new("123");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Number {
                value: 123,
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 3),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_symbol() {
        let mut lexer = Lexer::new(",.=$:\\()[]{}");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: ',',
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 1),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '.',
                span: Span {
                    line: 1,
                    column: 2,
                    range: (1, 2),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '=',
                span: Span {
                    line: 1,
                    column: 3,
                    range: (2, 3),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '$',
                span: Span {
                    line: 1,
                    column: 4,
                    range: (3, 4),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: ':',
                span: Span {
                    line: 1,
                    column: 5,
                    range: (4, 5),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '\\',
                span: Span {
                    line: 1,
                    column: 6,
                    range: (5, 6),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '(',
                span: Span {
                    line: 1,
                    column: 7,
                    range: (6, 7),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: ')',
                span: Span {
                    line: 1,
                    column: 8,
                    range: (7, 8),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '[',
                span: Span {
                    line: 1,
                    column: 9,
                    range: (8, 9),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: ']',
                span: Span {
                    line: 1,
                    column: 10,
                    range: (9, 10),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '{',
                span: Span {
                    line: 1,
                    column: 11,
                    range: (10, 11),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Symbol {
                value: '}',
                span: Span {
                    line: 1,
                    column: 12,
                    range: (11, 12),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_range() {
        let mut lexer = Lexer::new("0..=5");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Number {
                value: 0,
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 1),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::RangeOperator {
                mode: RangeOperator::Inclusive,
                span: Span {
                    line: 1,
                    column: 2,
                    range: (1, 4),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Number {
                value: 5,
                span: Span {
                    line: 1,
                    column: 5,
                    range: (4, 5),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);

        lexer = Lexer::new("0..<5");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Number {
                value: 0,
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 1),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::RangeOperator {
                mode: RangeOperator::Exclusive,
                span: Span {
                    line: 1,
                    column: 2,
                    range: (1, 4),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Number {
                value: 5,
                span: Span {
                    line: 1,
                    column: 5,
                    range: (4, 5),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_assignment_arrow() {
        let mut lexer = Lexer::new("->");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::AssignmentArrow {
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 2),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_application_arrow() {
        let mut lexer = Lexer::new("<|");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::ApplicationArrow {
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 2),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_match_arrow() {
        let mut lexer = Lexer::new("=>");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::MatchArrow {
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 2),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_newline() {
        let mut lexer = Lexer::new("\ntest\n\r\n\ntest");
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::Newline {
                span: Span {
                    line: 1,
                    column: 1,
                    range: (0, 1),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Identifier {
                name: "test",
                span: Span {
                    line: 2,
                    column: 1,
                    range: (1, 5),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Newline {
                span: Span {
                    line: 2,
                    column: 5,
                    range: (5, 6),
                },
            },
        );
        assert_eq!(
            lexer.next(),
            Token::Identifier {
                name: "test",
                span: Span {
                    line: 5,
                    column: 1,
                    range: (9, 13),
                },
            },
        );
        // assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_indentation() {
        todo!()

        /*
        #[test]
        fn test_indentation() {
            let mut lexer = Lexer::new("zero\n  one\n    two\n\n  one\n\n\n  one\nzero");
            lexer.lex().expect("failed to lex input");

            assert_eq!(lexer.next(), Token::Identifier("zero"));
            assert_eq!(lexer.next(), Token::Indent(2));
            assert_eq!(lexer.next(), Token::Identifier("one"));
            assert_eq!(lexer.next(), Token::Indent(4));
            assert_eq!(lexer.next(), Token::Identifier("two"));
            assert_eq!(lexer.next(), Token::Dedent(4));
            assert_eq!(lexer.next(), Token::Newline);
            assert_eq!(lexer.next(), Token::Identifier("one"));
            assert_eq!(lexer.next(), Token::Newline);
            assert_eq!(lexer.next(), Token::Identifier("one"));
            assert_eq!(lexer.next(), Token::Dedent(2));
            assert_eq!(lexer.next(), Token::Newline);
            assert_eq!(lexer.next(), Token::Identifier("zero"));
            assert_eq!(lexer.next(), Token::EOF);
        }
        */
    }
}
