mod lexer;
mod parser;
mod scanner;

use colored::Colorize;

fn main() {
    let example = r#"
# fruits -> ["apples", "bananas", "oranges"]

to_color -> case it of
  "apples"  => "red"
  "bananas" => "yellow"
  "oranges" => "orange"
  _         => "unknown"

colors ->
  each fruits do
    color -> to_color it
    "${it} are ${color}!"
  >> join it "\n"

"Apples are ${apple_color}!"
"#;

    let test = r#"
colors -> each fruits do write <| test "${it} is delicious!"
"#;

    let mut lexer = lexer::Lexer::new(example);
    lexer.lex().expect("failed to lex input");

    println!("--- Tokens:");

    let mut lexer_copy = lexer.clone();
    loop {
        let token = lexer_copy.next();
        let token_str = format!("{:?}", token);
        let token_str = match token {
            lexer::Token::EOF => token_str.bright_black(),
            lexer::Token::Identifier(_) => token_str.bright_blue(),
            lexer::Token::String(_) | lexer::Token::Regex(_) => token_str.bright_green(),
            lexer::Token::Number(_) => token_str.bright_magenta(),
            lexer::Token::Keyword(_) => token_str.bright_red(),
            lexer::Token::Newline | lexer::Token::Indent(_) | lexer::Token::Dedent(_) => {
                token_str.bright_yellow()
            }
            _ => token_str.bright_cyan(),
        };
        print!("{}\n", token_str);

        if token == lexer::Token::EOF {
            break;
        }
    }

    let mut parser = parser::Parser::new(lexer);
    let ast = parser.parse().expect("failed to parse input");

    println!("\n--- AST:");
    println!("{:#?}", ast);
}
