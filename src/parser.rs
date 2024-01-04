use crate::lexer::*;

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode<'a> {
    Prog {
        body: Vec<AstNode<'a>>,
    },
    FuncDecl {
        id: Box<AstNode<'a>>,
        params: Vec<&'a str>,
        body: Box<AstNode<'a>>,
    },
    Ident {
        name: &'a str,
    },
    StrLit {
        value: &'a str,
    },
    NumLit {
        value: f64,
    },
}

type PResult = Result<AstNode<'static>, PError>;

#[derive(Debug)]
pub enum PError {
    UnexpectedToken { expected: Token, found: Token },
    UnexpectedNode,
}

/// Tries a list of parsers and returns the result of the first one that
/// succeeds, or an error if none succeed.
macro_rules! one_of {
    ($self:ident : $first:ident $(, $rest:ident)*) => {
        {
            if let Ok(result) = $self.$first() {
                Ok(result)
            } $(else if let Ok(result) = $self.$rest() {
                Ok(result)
            })* else {
                Err(PError::UnexpectedNode)
            }
        }
    };
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
            return Err(PError::UnexpectedToken {
                expected: Token::EOF,
                found: self.lexer.peek(),
            });
        }

        Ok(program)
    }

    /// If the next token matches `token`, consumes it if `consume` is true.
    /// Otherwise, returns an error.
    fn expect(&mut self, token: Token, consume: bool) -> Result<(), PError> {
        if self.lexer.peek() == token {
            if consume {
                self.lexer.next();
            }

            Ok(())
        } else {
            Err(PError::UnexpectedToken {
                expected: token,
                found: self.lexer.peek(),
            })
        }
    }

    fn transform<T>(&mut self, cb: impl FnOnce(Token) -> Option<T>) -> Option<T> {
        match self.lexer.peek() {
            Token::EOF => None,
            token => match cb(token) {
                Some(output) => {
                    self.lexer.next();

                    Some(output)
                }
                None => None,
            },
        }
    }

    fn parse_program(&mut self) -> PResult {
        let mut body = Vec::new();

        loop {
            if let Ok(stmt) = self.parse_expr() {
                body.push(stmt);
            } else {
                break;
            }
        }

        Ok(AstNode::Prog { body })
    }

    fn parse_expr(&mut self) -> PResult {
        one_of!(self: parse_identifier, parse_string, parse_number)
    }

    fn parse_identifier(&mut self) -> PResult {
        match self.lexer.peek() {
            Token::Identifier(name) => {
                self.lexer.next();

                Ok(AstNode::Ident { name })
            }
            _ => Err(PError::UnexpectedToken {
                expected: Token::Identifier(""),
                found: self.lexer.peek(),
            }),
        }
    }

    fn parse_string(&mut self) -> PResult {
        match self.lexer.peek() {
            Token::String(value) => {
                self.lexer.next();

                Ok(AstNode::StrLit { value })
            }
            _ => Err(PError::UnexpectedToken {
                expected: Token::String(""),
                found: self.lexer.peek(),
            }),
        }
    }

    fn parse_number(&mut self) -> PResult {
        match self.lexer.peek() {
            Token::Number(whole) => {
                self.lexer.next();

                match self.lexer.peek() {
                    Token::Symbol('.') => {
                        self.lexer.next();

                        match self.lexer.peek() {
                            Token::Number(fraction) => {
                                self.lexer.next();

                                let number = format!("{}.{}", whole, fraction);
                                Ok(AstNode::NumLit {
                                    value: number.parse().unwrap(),
                                })
                            }
                            _ => Err(PError::UnexpectedToken {
                                expected: Token::Number(0),
                                found: self.lexer.peek(),
                            }),
                        }
                    }
                    _ => Ok(AstNode::NumLit {
                        value: whole as f64,
                    }),
                }
            }
            _ => Err(PError::UnexpectedToken {
                expected: Token::Number(0),
                found: self.lexer.peek(),
            }),
        }
    }
}
