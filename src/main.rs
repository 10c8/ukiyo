mod lexer;
mod scanner;

fn main() {
    let input = r#"
int -> re /[0-9]+/

name -> gen
age -> gen <| int <| stop ","
class -> gen
mantra -> gen <| temp 0.7
strength -> gen <| int <| stop ","

armor -> take valid_armor
weapon -> take valid_weapon

items -> gen <| temp 0.7 <| rep 3 >> join it ",\n"

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
  >> join "\n\n"

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
