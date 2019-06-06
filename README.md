<img src="assets/UpLabel.png" width="400" align="left"><br><br><br><br>
# UpLabel
A simple, automated, labeling tool for text classification and entity recognition.

#### Software Component Flow
---
<p><img src="assets/MLADS_Components.png" align="center" width="60%"></p>  

#### User Flow
---
<p><img src="assets/MLADS_UserFlow.png" align="center" width="60%"></p>

# Setup
1. Create a conda environment using environment.yml
2. Start 'Test - Pipeline' notebook

# Authors
Timm Walz (@torsion214)    
Martin Kayser (@maknotavailable)

# Glossary
| Word | Description |
|---|---|
| label | target category used by model and to be labelled |
| red(_id) | predicted label, respective numeric identifier |
| split | a subset of the data, to be distributed for manual labelling |

# Open Tasks
- [] Integrate with neanno frontend (https://github.com/timoklimmer/neanno).
- [] Support for Named Entity Recognition tasks.
- [] Support for Muli-Class Classification tasks.
- [] Active learning: targeted false positives
- [] Smart join: label quality score
- [] Smart load: data integrity validation 
