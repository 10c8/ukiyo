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
    Number(usize),
    Symbol(char),
    RangeOperator(RangeOperator),
    AssignmentArrow,
    MatchArrow,
    ExprApplicator,
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

pub struct Lexer {
    tokens: Vec<Token>,
}

impl Lexer {
    pub fn new(input: &str) -> Result<Self, LexError> {
        let mut scanner = Scanner::new(input);
        let mut tokens = Vec::new();

        let mut indent_stack = Vec::new();
        let mut tab_size = 0;

        while let Some(chr) = scanner.peek() {
            match chr {
                // Comment
                '#' => {
                    scanner.consume_while(|chr| *chr != '\n');
                    scanner.next();
                }
                // Newline (indentation)
                '\n' => {
                    scanner.consume_while(|chr| *chr == '\n');

                    let mut spaces = 0;

                    while let Some(chr) = scanner.peek() {
                        if *chr == ' ' {
                            spaces += 1;
                            scanner.next();
                        } else {
                            break;
                        }
                    }

                    if tab_size == 0 {
                        tab_size = spaces;
                    } else if spaces % tab_size != 0 {
                        return Err(LexError::UnexpectedIndent(
                            spaces,
                            scanner.line(),
                            scanner.column(),
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
                            scanner.line(),
                            scanner.column(),
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
                    scanner.next();
                }
                // Identifier
                chr if chr.is_ascii_lowercase() || *chr == '_' => {
                    let mut identifier = String::new();

                    identifier.push(*chr);
                    scanner.next();

                    while let Some(chr) = scanner.peek() {
                        if chr.is_ascii_lowercase() || *chr == '_' {
                            identifier.push(*chr);
                            scanner.next();
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
                    scanner.next();

                    while let Some(chr) = scanner.peek() {
                        if chr.is_digit(10) {
                            number.push(*chr);
                            scanner.next();
                        } else {
                            break;
                        }
                    }

                    let number = number
                        .parse::<usize>()
                        .map_err(|_| LexError::NumberOverflow(scanner.column(), scanner.line()))?;
                    tokens.push(Token::Number(number));
                }
                // String
                '"' => {
                    let mut string = String::new();
                    let is_multiline = scanner.try_consume_sequence("\"\"\"\n");

                    if is_multiline {
                        while let Some(_) = scanner.peek() {
                            if scanner.try_consume_sequence("\n\"\"\"") {
                                break;
                            } else {
                                string.push(*scanner.next().unwrap());
                            }
                        }
                    } else {
                        scanner.next();

                        while let Some(chr) = scanner.peek() {
                            match chr {
                                '\\' => {
                                    scanner.next();

                                    if let Some(chr) = scanner.peek() {
                                        match chr {
                                            '\\' | '"' => string.push(*chr),
                                            'n' => string.push('\n'),
                                            't' => string.push('\t'),
                                            _ => {
                                                return Err(LexError::UnexpectedChar(
                                                    *chr,
                                                    scanner.column(),
                                                    scanner.line(),
                                                ))
                                            }
                                        }

                                        scanner.next();
                                    } else {
                                        return Err(LexError::UnexpectedEOF(
                                            scanner.column(),
                                            scanner.line(),
                                        ));
                                    }
                                }
                                '"' => {
                                    scanner.next();
                                    break;
                                }
                                _ => {
                                    string.push(*chr);
                                    scanner.next();
                                }
                            }
                        }
                    }

                    tokens.push(Token::String(Box::leak(string.into_boxed_str())));
                }
                // Regex
                '/' => {
                    let mut regex = String::new();

                    scanner.next();

                    while let Some(chr) = scanner.peek() {
                        match chr {
                            '\\' => {
                                scanner.next();

                                if let Some(chr) = scanner.peek() {
                                    match chr {
                                        '\\' | '/' => regex.push(*chr),
                                        'n' => regex.push('\n'),
                                        't' => regex.push('\t'),
                                        _ => {
                                            return Err(LexError::UnexpectedChar(
                                                *chr,
                                                scanner.column(),
                                                scanner.line(),
                                            ))
                                        }
                                    }

                                    scanner.next();
                                } else {
                                    return Err(LexError::UnexpectedEOF(
                                        scanner.column(),
                                        scanner.line(),
                                    ));
                                }
                            }
                            '/' => {
                                scanner.next();
                                break;
                            }
                            _ => {
                                regex.push(*chr);
                                scanner.next();
                            }
                        }
                    }

                    tokens.push(Token::Regex(Box::leak(regex.into_boxed_str())));
                }
                // Assignment Arrow
                '-' => {
                    scanner.next();

                    if scanner.try_consume('>') {
                        tokens.push(Token::AssignmentArrow);
                    } else {
                        return Err(LexError::UnexpectedChar(
                            '-',
                            scanner.column(),
                            scanner.line(),
                        ));
                    }
                }
                // Match Arrow
                '=' => {
                    scanner.next();

                    if scanner.try_consume('>') {
                        tokens.push(Token::MatchArrow);
                    } else {
                        return Err(LexError::UnexpectedChar(
                            '=',
                            scanner.column(),
                            scanner.line(),
                        ));
                    }
                }
                // Symbol / Range Operator
                '.' => {
                    scanner.next();

                    if scanner.try_consume('.') {
                        if scanner.try_consume('=') {
                            tokens.push(Token::RangeOperator(RangeOperator::Inclusive));
                        } else if scanner.try_consume('<') {
                            tokens.push(Token::RangeOperator(RangeOperator::Exclusive));
                        } else {
                            if let Some(invalid) = scanner.peek() {
                                return Err(LexError::UnexpectedChar(
                                    *invalid,
                                    scanner.column(),
                                    scanner.line(),
                                ));
                            } else {
                                return Err(LexError::UnexpectedEOF(
                                    scanner.column(),
                                    scanner.line(),
                                ));
                            }
                        }
                    } else {
                        tokens.push(Token::Symbol('.'));
                    }
                }
                // Expression Applicator
                '<' => {
                    scanner.next();

                    if scanner.try_consume('|') {
                        tokens.push(Token::ExprApplicator);
                    } else {
                        return Err(LexError::UnexpectedChar(
                            '<',
                            scanner.column(),
                            scanner.line(),
                        ));
                    }
                }
                // Unknown
                _ => {
                    return Err(LexError::UnexpectedChar(
                        *chr,
                        scanner.column(),
                        scanner.line(),
                    ));
                }
            };
        }

        tokens.reverse();

        Ok(Lexer { tokens })
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
        )
        .expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Identifier("test"));
        assert_eq!(lexer.next(), Token::AssignmentArrow);
        assert_eq!(lexer.next(), Token::String("test"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_identifier() {
        let mut lexer = Lexer::new("hello").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Identifier("hello"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_keyword() {
        let mut lexer = Lexer::new("case").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Keyword(Keyword::Case));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_string() {
        let mut lexer = Lexer::new(r#""hello world\n""#).expect("failed to lex input");

        assert_eq!(lexer.next(), Token::String("hello world\n"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_escaped_string() {
        let mut lexer = Lexer::new(r#""hello \"world\"\n"#).expect("failed to lex input");

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
        )
        .expect("failed to lex input");

        assert_eq!(
            lexer.next(),
            Token::String("hello \"world\"\nthis is cool!")
        );
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_regex() {
        let mut lexer = Lexer::new(r#"/[a-zA-Z]+/"#).expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Regex("[a-zA-Z]+"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_number() {
        let mut lexer = Lexer::new("123").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Number(123));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_symbol() {
        let mut lexer = Lexer::new(".").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Symbol('.'));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_range() {
        let mut lexer = Lexer::new("0..=5").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Number(0));
        assert_eq!(lexer.next(), Token::RangeOperator(RangeOperator::Inclusive));
        assert_eq!(lexer.next(), Token::Number(5));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_assignment_arrow() {
        let mut lexer = Lexer::new("->").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::AssignmentArrow);
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_match_arrow() {
        let mut lexer = Lexer::new("=>").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::MatchArrow);
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_expr_applicator() {
        let mut lexer = Lexer::new("<|").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::ExprApplicator);
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_newline() {
        let mut lexer = Lexer::new("\ntest\n\n\ntest").expect("failed to lex input");

        assert_eq!(lexer.next(), Token::Newline);
        assert_eq!(lexer.next(), Token::Identifier("test"));
        assert_eq!(lexer.next(), Token::Newline);
        assert_eq!(lexer.next(), Token::Identifier("test"));
        assert_eq!(lexer.next(), Token::EOF);
    }

    #[test]
    fn test_indentation() {
        let mut lexer = Lexer::new("zero\n  one\n    two\n\n  one\n\n\n  one\nzero")
            .expect("failed to lex input");

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
