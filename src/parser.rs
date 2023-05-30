use crate::lexer::*;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(&'static str, Token, usize),
    UnexpectedEOF(&'static str, usize),
    NumberParseError(String, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Program {
        children: Vec<Node>,
    },
    Block {
        children: Vec<Node>,
    },
    ExpressionDefinition {
        name: String,
        expression: Box<Node>,
    },
    ExpressionCall {
        name: String,
        arguments: Vec<Node>,
    },
    ExpressionApplication {
        left: Box<Node>,
        right: Box<Node>,
    },
    ScopeBinding {
        left: Box<Node>,
        right: Box<Node>,
    },
    Case {
        target: Box<Node>,
        children: Vec<Node>,
    },
    CaseBlock {
        children: Vec<Node>,
    },
    CaseMatch {
        left: Box<Node>,
        right: Box<Node>,
    },
    Each {
        target: Box<Node>,
        children: Vec<Node>,
    },
    Identifier(&'static str),
    String(&'static str),
    Integer(u64),
    Float(f64),
    Range {
        is_exclusive: bool,
        start: Box<Node>,
        end: Box<Node>,
    },
}

pub struct Parser {
    lexer: Lexer,
}

impl Parser {
    pub fn new(lexer: Lexer) -> Self {
        Parser { lexer }
    }

    pub fn parse(&mut self) -> Result<Node, ParseError> {
        let program = self.program()?;

        if self.lexer.peek() != Token::EOF {
            return Err(ParseError::UnexpectedToken(
                "program",
                self.lexer.peek(),
                self.lexer.cursor(),
            ));
        }

        Ok(program)
    }

    /// Consumes the next token if it matches `token`.
    /// Otherwise, returns an error.
    fn require(&mut self, token: Token) -> Result<(), ParseError> {
        if self.lexer.peek() == token {
            self.lexer.next();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(
                "require",
                self.lexer.peek(),
                self.lexer.cursor(),
            ))
        }
    }

    // Node parsers
    fn program(&mut self) -> Result<Node, ParseError> {
        // program = SOI '\n'* stmt* EOI

        let mut children = Vec::new();

        loop {
            match self.lexer.peek() {
                Token::EOF => break,
                Token::Newline => {
                    self.lexer.next();
                }
                _ => children.push(self.statement()?),
            }
        }

        Ok(Node::Program { children })
    }

    fn statement(&mut self) -> Result<Node, ParseError> {
        // stmt = exprdef | case | each | term

        let result = match self.lexer.peek() {
            Token::Identifier(_) => match self.expression_definition() {
                Ok(node) => node,
                Err(_) => self.term()?,
            },
            Token::String(_) | Token::Number(_) => self.term()?,
            Token::Keyword(Keyword::Case) => todo!(), // self.case()?,
            Token::Keyword(Keyword::Each) => todo!(), // self.each()?,
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "statement",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        Ok(result)
    }

    fn expression_definition(&mut self) -> Result<Node, ParseError> {
        // exprdef = identifier '->' expr

        let name = match self.lexer.peek() {
            Token::Identifier(name) => {
                self.lexer.next();
                name
            }
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "exprdef",
                    Token::EOF,
                    self.lexer.cursor(),
                ))
            }
        };

        self.require(Token::AssignmentArrow)?;

        let expression = self.expression()?;

        Ok(Node::ExpressionDefinition {
            name: name.to_string(),
            expression: Box::new(expression),
        })
    }

    fn expression(&mut self) -> Result<Node, ParseError> {
        // expr = term expr_tail
        // expr_tail = ('<|' expr | ident term*)*

        let left = self.term()?;

        println!("expr: {:?}", left);
        println!("peek: {:?}", self.lexer.peek());

        let mut current = left;

        loop {
            match self.lexer.peek() {
                Token::ApplicationArrow => {
                    self.lexer.next();

                    let right = self.expression()?;

                    current = Node::ExpressionApplication {
                        left: Box::new(current),
                        right: Box::new(right),
                    };
                }
                Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                    let mut arguments = Vec::new();

                    loop {
                        println!("internal peek: {:?}", self.lexer.peek());

                        match self.lexer.peek() {
                            Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                                arguments.push(self.term()?)
                            }
                            _ => break,
                        }
                    }

                    current = Node::ExpressionCall {
                        name: match current {
                            Node::Identifier(name) => name.to_string(),
                            _ => {
                                return Err(ParseError::UnexpectedToken(
                                    "expression",
                                    self.lexer.peek(),
                                    self.lexer.cursor(),
                                ))
                            }
                        },
                        arguments,
                    };
                }
                _ => break,
            }
        }

        Ok(current)
    }

    fn term(&mut self) -> Result<Node, ParseError> {
        let result = match self.lexer.next() {
            Token::Identifier(name) => Node::Identifier(name),
            Token::String(string) => Node::String(string),
            Token::Number(num) => match self.lexer.peek() {
                Token::Symbol('.') => {
                    self.lexer.next();

                    let next = self.lexer.next();

                    match next {
                        Token::Number(frac) => {
                            let number = format!("{}.{}", num, frac);
                            Node::Float(number.parse().map_err(|_| {
                                ParseError::NumberParseError(number, self.lexer.cursor())
                            })?)
                        }
                        _ => {
                            return Err(ParseError::UnexpectedToken(
                                "number",
                                next,
                                self.lexer.cursor(),
                            ))
                        }
                    }
                }
                Token::RangeOperator(op) => {
                    return Err(ParseError::UnexpectedToken(
                        "number",
                        Token::RangeOperator(op),
                        self.lexer.cursor(),
                    ))
                }
                _ => Node::Integer(num),
            },
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "term",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        Ok(result)
    }

    fn range(&mut self) -> Result<Node, ParseError> {
        self.lexer.next();

        let start = match self.lexer.next() {
            Token::Number(num) => Node::Integer(num),
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        let op = match self.lexer.next() {
            Token::RangeOperator(op) => op,
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        let end = match self.lexer.next() {
            Token::Number(num) => Node::Integer(num),
            _ => {
                return Err(ParseError::UnexpectedToken(
                    "range",
                    self.lexer.peek(),
                    self.lexer.cursor(),
                ))
            }
        };

        Ok(Node::Range {
            is_exclusive: op == RangeOperator::Exclusive,
            start: Box::new(start),
            end: Box::new(end),
        })
    }
}
