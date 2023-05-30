mod lexer;
mod parser;
mod scanner;

fn main() {
    let input = r#"
# int -> re /[0-9]+/

name -> gen
age -> gen <| int <| stop ","
class -> gen
mantra -> gen <| temp 0.7
strength -> gen <| int <| stop ","

armor -> take valid_armor
weapon -> take valid_weapon

items -> gen <| temp 0.7 <| rep 3 # >> join it ",\n"

examples ->
  each example_list
    entities -> each it.entities " - ${it.name}: ${it.time}\n"

    """
    Sentence: ${it.sentence}
    Entities and dates:
    ${entities}
    Reasoning: ${it.reasoning}
    Anachronism: ${it.answer}
    """
  # >> join "\n\n"

"""
The following is a character profile for an RPG game in JSON format:
```json
{
  "id": "${id}",
  "description": "${description}",
  "name": "${name}",
  "age": ${age},
  "class": "${class}",
  "mantra": "${mantra}",
  "strength": ${strength},
  "armor": "${armor}",
  "weapon": "${weapon}",
  "items": [
    ${items}
  ]
}
"""
"#;

    let example = r#"
color -> case it of
  "apple"  => "red"
  "banana" => "yellow"
  "orange" => "orange"
  _ => "unknown"

apple_color -> color "apple"

"Apples are ${apple_color}!"
"#;

    let mut lexer = lexer::Lexer::new(example);
    lexer.lex().expect("failed to lex input");

    println!("--- Tokens:");

    let mut lexer_copy = lexer.clone();
    loop {
        let token = lexer_copy.next();
        println!("{:?}", token);

        if token == lexer::Token::EOF {
            break;
        }
    }

    let mut parser = parser::Parser::new(lexer);
    let ast = parser.parse().expect("failed to parse input");

    println!("\n--- AST:");
    println!("{:#?}", ast);
}
