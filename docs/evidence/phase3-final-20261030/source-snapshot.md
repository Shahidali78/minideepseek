# Frozen evaluation source

`frozen-source/` contains the exact source/configuration/requirements bytes hashed before the held-out run. Its relative paths match the public manifest's `code_sha256` entries. The source snapshot contains no model files, datasets or raw completions. Restore it into a separate checkout when auditing the historical evaluator; do not replace current working files casually.

The current publication version repairs the final runner's source-copy path so future freezes always copy the runner into their snapshot directory. In the recorded run, that file was manually copied byte-for-byte into the intended snapshot path before the evaluation completed; its recorded hash matches. This repair changes neither model loading, generation, scoring nor the completed outcomes. The frozen snapshot, rather than a later publication commit, identifies the executed experiment.
