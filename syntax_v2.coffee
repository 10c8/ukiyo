fruits -> ["apples", "bananas", "oranges", "limes"]
flavors -> {
  "fruit"     = ["sweet", "sour", "bitter"],
  "vegetable" = ["bitter", "salty", "umami"],
  "meat"      = ["salty", "umami", "sweet"],
}

get_color fruit ->
  color -> case fruit of
    "apples"            => "red"
    "bananas" | "limes" => "yellow"
    "oranges"           => "orange"
    _                   => gen %% { "ctx" = ["give me the fruit color"], "temp" = 0.3 }

  $"{fruit} are {color fruit}"

fruit_colors ->
  case len fruits of
    1 => head fruits <| get_color
    _ =>
      start -> (init fruits |f| get_color f) <| join ", "
      end   -> last fruits <| get_color

      $"{start} and {end}"

fruit_flavors ->
  ff -> flavors :: "fruit"

  case len ff of
    1 => head ff
    _ => $"{init ff <| join ", "} and {tail ff}"

$@"
Fruits have many colors: {fruit_colors}.
Oh, and by the way, they can also display many flavors, such as {fruit_flavors}.
"@
