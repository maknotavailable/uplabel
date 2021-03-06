<img src="assets/UpLabel.png" width="400" align="left"><br><br><br><br>
# UpLabel
UpLabel is a lightweight, Python-based and modular tool which serves to support your machine learning tasks by making the data labeling process more efficient, automated and structured. In the current version, the tool is mainly focused on text classification tasks.

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
Timm Walz (@nonstoptimm)    
Martin Kayser (@maknotavailable)

# Glossary
| Word | Description |
|---|---|
| label | target category used by model and to be labeled |
| pred(\_id) | predicted label, respective numeric identifier |
| split | a subset of the data, to be distributed for manual labelling |

# Open Tasks
## In Progress
- [ ] Host as a service in Azure (via FA)
- [ ] Improve complexity calculation
## TODO
- [ ] Integrate with neanno frontend (https://github.com/timoklimmer/neanno).
- [ ] Support for Named Entity Recognition tasks.
- [ ] Support for Muli-Class Classification tasks.
- [ ] Active learning: targeted false positives
- [ ] Smart join: label quality score
- [ ] Smart load: data integrity validation 
- [ ] Auto-create labeling documentation
