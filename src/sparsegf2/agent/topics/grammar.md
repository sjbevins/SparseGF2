# The call grammar

Structured arguments are strings that look like a Python call. The server parses them; nothing is evaluated.

```text
item   := name [ "(" args ")" ] [ "@" checkpoints ]
args   := arg ("," arg)*       arg := value | key "=" value
value  := integer | number | true | false | range | list | word | "quoted string"
range  := int ":" int [ ":" int ]       (Python slices: 0:8 is 0..7)
list   := "[" value ("," value)* "]"
checkpoints := final | every_layer | int (a stride) | list of layers
```

Rules:

- Names come from the library. Use `help_list` for every name with its signature (probes, observables, code properties, metrics, graph families, edge weight families).
- Arguments are positional first, then `key=value`. Positional arguments map onto the parameters in the order `help_list` shows.
- `@checkpoints` is only accepted on probes, and gives that probe its own checkpoints.
- Words are Pauli products and names: `Z0*Z1`, `XXZZ`, `random`, `power_law`.

Examples:

| where | examples |
|---|---|
| probes | `code_dimension`, `entropy(0:8)`, `mutual_information(a=0:4, b=8:12)@every_layer`, `erasure_first_failure(samples=512)@[64,128,256]` |
| observables | `entropy(0:4)`, `expectation(Z0*Z1)`, `code_dimension(reference=16:20)` |
| edge weights | `power_law(alpha=2.0)` |
| sweep grids | `measurement_rate=[0.05,0.1,0.15]; num_system=[64,128]` |

Errors name the position. Two common ones: `unknown argument 'regon' at column 9; entropy takes (region)` (a misspelt argument name: use the names in the message), and an unknown name, which lists the close matches.
