mod lexer;
mod parser;
mod scanner;

fn main() {
    let example = r#"
greet name -> $"Hello, {name}!"
greet "world"
list -> [1, "2", three]
"#;

    let mut lexer = lexer::Lexer::new(example);
    lexer.lex().expect("failed to lex input");

    let mut lexer_copy = lexer.clone();

    let mut parser = parser::Parser::new(lexer);
    let ast = parser.parse();
    if let Err(err) = ast {
        println!("--- Tokens:");
        loop {
            let token = lexer_copy.next();
            print!("{:#?}\n", token);

            if token == lexer::Token::EOF {
                break;
            }
        }

        println!("\n{}", parser.display_error(err).unwrap());
        return;
    }

    println!("\n--- AST:");
    println!("{:#?}", ast.unwrap());
}
