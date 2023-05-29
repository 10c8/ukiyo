mod lexer;
mod scanner;

fn main() {
    let input = r#"
list -> each 1..=3 do
  suffix -> case it of
    1 => "st"
    2 => "nd"
    3 => "rd"
    _ => "th"

  thing -> gen <| temp 0.6

  "${it}${suffix} thing: ${thing}."

"""
Here's a list of things similar to ${input}:
${list}
"""
"#;

    let mut lexer = lexer::Lexer::new(input).expect("failed to lex input");

    println!("---\nTokens:");

    loop {
        let token = lexer.next();
        println!("{:?}", token);

        if token == lexer::Token::EOF {
            break;
        }
    }
}
