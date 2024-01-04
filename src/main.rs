mod lexer;
mod parser;
mod scanner;

fn main() {
    let example = r#"
    hello 8 "Hello, world!"
"#
    .trim();

    let mut lexer = lexer::Lexer::new(example);
    lexer.lex().expect("failed to lex input");

    println!("--- Tokens:");

    let mut lexer_copy = lexer.clone();
    loop {
        let token = lexer_copy.next();
        print!("{:?}\n", token);

        if token == lexer::Token::EOF {
            break;
        }
    }

    let mut parser = parser::Parser::new(lexer);
    let ast = parser.parse().expect("failed to parse input");

    println!("\n--- AST:");
    println!("{:#?}", ast);
}
