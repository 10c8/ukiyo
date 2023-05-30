use crate::scanner::*;

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
    Identifier(&'static str),
    Keyword(Keyword),
    String(&'static str),
    Regex(&'static str),
    Number(u64),
    Symbol(char),
    RangeOperator(RangeOperator),
    AssignmentArrow,
    ScopeBindingArrow,
    MatchArrow,
    ApplicationArrow,
    Newline,
    Indent(usize),
    Dedent(usize),
    EOF,
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
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let scanner = Scanner::new(input);
        let tokens = Vec::new();

        Self { scanner, tokens }
    }

    pub fn lex(&mut self) -> Result<(), LexError> {
        let mut tokens = Vec::new();

        let mut indent_stack = Vec::new();
        let mut tab_size = 0;

        while let Some(chr) = self.scanner.peek() {
            match chr {
                // Comment
                '#' => {
                    self.scanner.consume_while(|chr| *chr != '\n');
                    self.scanner.next();
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

                    if tab_size == 0 {
                        tab_size = spaces;
                    } else if spaces % tab_size != 0 {
                        return Err(LexError::UnexpectedIndent(
                            spaces,
                            self.scanner.line(),
                            self.scanner.column(),
                        ));
                    }

                    if tab_size == 0 {
                        tokens.push(Token::Newline);
                        continue;
                    }

                    if spaces == 0 && !indent_stack.is_empty() {
                        while let Some(last_indent) = indent_stack.last() {
                            tokens.push(Token::Dedent(*last_indent));
                            indent_stack.pop();
                        }
                        continue;
                    }

                    let indent = spaces / tab_size;
                    let last_indent = *indent_stack.last().unwrap_or(&0);

                    if indent > last_indent + 1 {
                        return Err(LexError::UnexpectedIndent(
                            spaces,
                            self.scanner.line(),
                            self.scanner.column(),
                        ));
                    } else if indent > last_indent {
                        indent_stack.push(indent);
                        tokens.push(Token::Indent(indent));
                    } else if indent < last_indent {
                        while let Some(last_indent) = indent_stack.last() {
                            if indent < *last_indent {
                                tokens.push(Token::Dedent(*last_indent));
                                indent_stack.pop();
                            } else {
                                break;
                            }
                        }
                    } else {
                        tokens.push(Token::Newline);
                    }
                }
                // Whitespace
                chr if chr.is_whitespace() => {
                    self.scanner.next();
                }
                // Identifier
                chr if chr.is_ascii_lowercase() || *chr == '_' => {
                    let mut identifier = String::new();

                    identifier.push(*chr);
                    self.scanner.next();

                    while let Some(chr) = self.scanner.peek() {
                        if chr.is_ascii_lowercase() || *chr == '_' {
                            identifier.push(*chr);
                            self.scanner.next();
                        } else {
                            break;
                        }
                    }

                    let token = match identifier.as_str() {
                        "case" => Token::Keyword(Keyword::Case),
                        "do" => Token::Keyword(Keyword::Do),
                        "each" => Token::Keyword(Keyword::Each),
                        "of" => Token::Keyword(Keyword::Of),
                        _ => Token::Identifier(Box::leak(identifier.into_boxed_str())),
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

                    let number = number.parse::<u64>().map_err(|_| {
                        LexError::NumberOverflow(self.scanner.column(), self.scanner.line())
                    })?;
                    tokens.push(Token::Number(number));
                }
                // String
                '"' => {
                    let mut string = String::new();
                    let is_multiline = self.scanner.try_consume_sequence("\"\"\"\n");

                    if is_multiline {
                        while let Some(_) = self.scanner.peek() {
                            if self.scanner.try_consume('\n') {
                                let mut spaces = 0;
                                let mut indent = 0;

                                let last_indent = *indent_stack.last().unwrap_or(&0);

                                if last_indent > 0 {
                                    while let Some(chr) = self.scanner.peek() {
                                        if *chr == ' ' {
                                            spaces += 1;
                                            self.scanner.next();
                                        } else {
                                            break;
                                        }
                                    }

                                    if spaces % tab_size != 0 {
                                        return Err(LexError::UnexpectedIndent(
                                            spaces,
                                            self.scanner.line(),
                                            self.scanner.column(),
                                        ));
                                    }

                                    indent = spaces / tab_size;
                                }

                                if self.scanner.try_consume_sequence("\"\"\"") {
                                    if tab_size > 0 {
                                        if indent != last_indent {
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
                                string.push(*self.scanner.next().unwrap());
                            }
                        }
                    } else {
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
                    }

                    tokens.push(Token::String(Box::leak(string.into_boxed_str())));
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

                    tokens.push(Token::Regex(Box::leak(regex.into_boxed_str())));
                }
                // Assignment Arrow
                '-' => {
                    self.scanner.next();

                    if self.scanner.try_consume('>') {
                        tokens.push(Token::AssignmentArrow);
                    } else {
                        return Err(LexError::UnexpectedChar(
                            '-',
                            self.scanner.column(),
                            self.scanner.line(),
                        ));
                    }
                }
                // Match Arrow
                '=' => {
                    self.scanner.next();

                    if self.scanner.try_consume('>') {
                        tokens.push(Token::MatchArrow);
                    } else {
                        return Err(LexError::UnexpectedChar(
                            '=',
                            self.scanner.column(),
                            self.scanner.line(),
                        ));
                    }
                }
                // Scope Binding Arrow
                '>' => {
                    self.scanner.next();

                    if self.scanner.try_consume('>') {
                        tokens.push(Token::ScopeBindingArrow);
                    } else {
                        return Err(LexError::UnexpectedChar(
                            '>',
                            self.scanner.column(),
                            self.scanner.line(),
                        ));
                    }
                }
                // Symbol / Range Operator
                '.' => {
                    self.scanner.next();

                    if self.scanner.try_consume('.') {
                        if self.scanner.try_consume('=') {
                            tokens.push(Token::RangeOperator(RangeOperator::Inclusive));
                        } else if self.scanner.try_consume('<') {
                            tokens.push(Token::RangeOperator(RangeOperator::Exclusive));
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
                        tokens.push(Token::Symbol('.'));
                    }
                }
                // Expression Applicator
                '<' => {
                    self.scanner.next();

                    if self.scanner.try_consume('|') {
                        tokens.push(Token::ApplicationArrow);
                    } else {
                        return Err(LexError::UnexpectedChar(
                            '<',
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

        tokens.reverse();

        self.tokens = tokens;

        Ok(())
    }

    pub fn cursor(&self) -> usize {
        self.tokens.len()
    }

    pub fn peek(&mut self) -> Token {
        self.tokens.last().copied().unwrap_or(Token::EOF)
    }

    pub fn next(&mut self) -> Token {
        self.tokens.pop().unwrap_or(Token::EOF)
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

        assert_eq!(lexer.next(), Token::Identifier("test"));
        assert_eq!(lexer.next(), Token::AssignmentArrow);
        assert_eq!(lexer.next(), Token::String("test"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_identifier() {
        let mut lexer = Lexer::new("hello");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Identifier("hello"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_keyword() {
        let mut lexer = Lexer::new("case");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Keyword(Keyword::Case));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_string() {
        let mut lexer = Lexer::new(r#""hello world\n""#);
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::String("hello world\n"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_escaped_string() {
        let mut lexer = Lexer::new(r#""hello \"world\"\n"#);
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::String("hello \"world\"\n"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_multiline_string() {
        let mut lexer = Lexer::new(
            r#""""
hello "world"
this is cool!
""""#,
        );
        lexer.lex().expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::String("hello \"world\"\nthis is cool!")
        );
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_regex() {
        let mut lexer = Lexer::new(r#"/[a-zA-Z]+/"#);
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Regex("[a-zA-Z]+"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_number() {
        let mut lexer = Lexer::new("123");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Number(123));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_symbol() {
        let mut lexer = Lexer::new(".");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Symbol('.'));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_range() {
        let mut lexer = Lexer::new("0..=5");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Number(0));
        assert_eq!(lexer.next(), Token::RangeOperator(RangeOperator::Inclusive));
        assert_eq!(lexer.next(), Token::Number(5));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_assignment_arrow() {
        let mut lexer = Lexer::new("->");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::AssignmentArrow);
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_match_arrow() {
        let mut lexer = Lexer::new("=>");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::MatchArrow);
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_scope_binding_arrow() {
        let mut lexer = Lexer::new(">>");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::ScopeBindingArrow);
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_expr_applicator() {
        let mut lexer = Lexer::new("<|");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::ApplicationArrow);
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_newline() {
        let mut lexer = Lexer::new("\ntest\n\n\ntest");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Newline);
        assert_eq!(lexer.next(), Token::Identifier("test"));
        assert_eq!(lexer.next(), Token::Newline);
        assert_eq!(lexer.next(), Token::Identifier("test"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_indentation() {
        let mut lexer = Lexer::new("zero\n  one\n    two\n\n  one\n\n\n  one\nzero");
        lexer.lex().expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Identifier("zero"));
        assert_eq!(lexer.next(), Token::Indent(1));
        assert_eq!(lexer.next(), Token::Identifier("one"));
        assert_eq!(lexer.next(), Token::Indent(2));
        assert_eq!(lexer.next(), Token::Identifier("two"));
        assert_eq!(lexer.next(), Token::Dedent(2));
        assert_eq!(lexer.next(), Token::Identifier("one"));
        assert_eq!(lexer.next(), Token::Newline);
        assert_eq!(lexer.next(), Token::Identifier("one"));
        assert_eq!(lexer.next(), Token::Dedent(1));
        assert_eq!(lexer.next(), Token::Identifier("zero"));
        assert_eq!(lexer.next(), Token::EOF);
    }
}
