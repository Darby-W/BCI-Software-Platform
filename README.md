BCI-Software-Platform

A modular Brain-Computer Interface (BCI) software platform for offline experiment management, data processing, and algorithm evaluation.

This project is developed as a research-oriented platform with a clear separation of:

- Data Management
- Algorithm Module (Plugin-based)
- Experiment Pipeline
- Reproducible Results

---

1.Clone the Repository
```bash
git clone https://github.com/Darby-W/BCI-Software-Platform.git
cd BCI-Software-Platform

---

2.Install Dependencies
We recommend using a virtual environment (optional but recommended):
pip install -r requirements.txt
Currently required:numpy

---

3.Run the MVP
python run_mvp.py
If successful, you should see:
[OK] run_id=...
[OK] saved to: results/...

---

4.Current Status

✔ MVP runnable
✔ Unified algorithm interface
✔ Structured experiment output
✔ Git-based collaborative development

Next steps:
Integrate real data management
Support multiple algorithm plugins
Add experiment comparison tools

