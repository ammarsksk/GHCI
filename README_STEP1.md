# txcat (Step 1: Repo scaffold)

This is the minimal scaffold for the AI transaction categorisation project.

## Structure
```
txcat/
├── config/
│   └── taxonomy.yaml      # edit categories/keywords here
├── data/                  # datasets will live here
├── models/                # trained artifacts later
└── src/                   # source code (added in next steps)
```

## What to do now

1) Move this folder somewhere on your machine and open a terminal in the **txcat** directory.

2) (Optional) Initialize git:
```
git init
git add .
git commit -m "Step 1 scaffold with taxonomy.yaml"
```

3) Open `config/taxonomy.yaml` and review categories/keywords. You can change them at any time; the rules layer will use this file without retraining.
