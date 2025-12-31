# Multi-Robot Safe Mode, Threshold Ablation & Risk–Length λ-Sweep Experiments

## 1️⃣ Πλαίσιο Προβλήματος
Σε multi-robot navigation με αβεβαιότητα αισθητήρων, τα ρομπότ ενδέχεται να κινούνται σε περιοχές υψηλού ρίσκου (π.χ. υψηλό drift / pose uncertainty).  
Για αυτό υλοποιήθηκε μηχανισμός:

- **Self-Trust Estimation S(t)** (0–1)
- **Safe Mode Policy Switching**
- **Risk-Aware Task Allocation**

---

## 2️⃣ Safe Mode Policy
Η βασική ιδέα:
- Αν `S ≥ τ` → κανονική λειτουργία (**NORMAL_POLICY**)
- Αν `S < τ` → ασφαλής λειτουργία (**SAFE_MODE_POLICY**)

Στόχος SAFE mode:
- μείωση συνολικού ρίσκου
- μείωση μέγιστου κινδύνου ανά στόχο
- ακόμη και με μεγαλύτερο κόστος απόστασης

---

## 3️⃣ Threshold Ablation Study
**Αρχείο εκτέλεσης:**
# Multi-Robot Safe Mode, Threshold Ablation & Risk–Length λ-Sweep Experiments

## 1️⃣ Πλαίσιο Προβλήματος
Σε multi-robot navigation με αβεβαιότητα αισθητήρων, τα ρομπότ ενδέχεται να κινούνται σε περιοχές υψηλού ρίσκου (π.χ. υψηλό drift / pose uncertainty).  
Για αυτό υλοποιήθηκε μηχανισμός:

- **Self-Trust Estimation S(t)** (0–1)
- **Safe Mode Policy Switching**
- **Risk-Aware Task Allocation**

---

## 2️⃣ Safe Mode Policy
Η βασική ιδέα:
- Αν `S ≥ τ` → κανονική λειτουργία (**NORMAL_POLICY**)
- Αν `S < τ` → ασφαλής λειτουργία (**SAFE_MODE_POLICY**)

Στόχος SAFE mode:
- μείωση συνολικού ρίσκου
- μείωση μέγιστου κινδύνου ανά στόχο
- ακόμη και με μεγαλύτερο κόστος απόστασης

---

## 3️⃣ Threshold Ablation Study
**Αρχείο εκτέλεσης:**
# Multi-Robot Safe Mode, Threshold Ablation & Risk–Length λ-Sweep Experiments

## 1️⃣ Πλαίσιο Προβλήματος
Σε multi-robot navigation με αβεβαιότητα αισθητήρων, τα ρομπότ ενδέχεται να κινούνται σε περιοχές υψηλού ρίσκου (π.χ. υψηλό drift / pose uncertainty).  
Για αυτό υλοποιήθηκε μηχανισμός:

- **Self-Trust Estimation S(t)** (0–1)
- **Safe Mode Policy Switching**
- **Risk-Aware Task Allocation**

---

## 2️⃣ Safe Mode Policy
Η βασική ιδέα:
- Αν `S ≥ τ` → κανονική λειτουργία (**NORMAL_POLICY**)
- Αν `S < τ` → ασφαλής λειτουργία (**SAFE_MODE_POLICY**)

Στόχος SAFE mode:
- μείωση συνολικού ρίσκου
- μείωση μέγιστου κινδύνου ανά στόχο
- ακόμη και με μεγαλύτερο κόστος απόστασης

---

## 3️⃣ Threshold Ablation Study
**Αρχείο εκτέλεσης:**
python3 safe_mode_threshold_ablation.py


Δοκιμάσαμε τιμές κατωφλίου:


τ = 0.50, 0.60, 0.70, 0.80



### ⭐ Κύριο Εύρημα
Για κάθε threshold παρατηρήθηκε:
- τα πρώτα ≈5 βήματα → NORMAL policy  
- όταν το S έπεσε κάτω από το threshold → ενεργοποιήθηκε SAFE mode  
- σαφής trade-off:

| Policy | Distance | Risk | Max Risk | Total Cost |
|--------|---------:|-----:|---------:|-----------:|
| NORMAL | 1.1 | 1.2 | 0.9 | 4.9 |
| SAFE | 4.0 | 0.4 | 0.2 | 10.4 |

👉 Άρα:
- **SAFE mode = μειώνει δραματικά το ρίσκο**
- **αλλά αυξάνει το κόστος & την απόσταση**
- διαφορετικά thresholds ελέγχουν ΠΟΤΕ ενεργοποιείται

Αυτό δίνει ξεκάθαρο engineering & scientific control mechanism.

---

## 4️⃣ λ-Sweep (Risk vs Distance Trade-off)

Τρέχεις:

### ⭐ Κύριο Εύρημα
Για κάθε threshold παρατηρήθηκε:
- τα πρώτα ≈5 βήματα → NORMAL policy  
- όταν το S έπεσε κάτω από το threshold → ενεργοποιήθηκε SAFE mode  
- σαφής trade-off:

| Policy | Distance | Risk | Max Risk | Total Cost |
|--------|---------:|-----:|---------:|-----------:|
| NORMAL | 1.1 | 1.2 | 0.9 | 4.9 |
| SAFE | 4.0 | 0.4 | 0.2 | 10.4 |

👉 Άρα:
- **SAFE mode = μειώνει δραματικά το ρίσκο**
- **αλλά αυξάνει το κόστος & την απόσταση**
- διαφορετικά thresholds ελέγχουν ΠΟΤΕ ενεργοποιείται

Αυτό δίνει ξεκάθαρο engineering & scientific control mechanism.

---

## 4️⃣ λ-Sweep (Risk vs Distance Trade-off)

Τρέχεις:
python3 analyze_lambda_sweep_risk_length.py


Χρησιμοποιείται κόστος:
J = Distance + λ · Risk


Τιμές που ελέγχθηκαν:
λ = 0, 0.5, 1, 2, 4, 8, 16


### ⭐ Κύριο Εύρημα

| λ | Distance | Risk | Max Risk | Assignment | Cost |
|----|---------:|------:|----------:|-----------|------:|
| 0.0 | 1.1 | 1.2 | 0.9 | riskier targets | 1.1 |
| 0.5 | 1.1 | 1.2 | 0.9 | same | 1.7 |
| 1.0 | 1.1 | 1.2 | 0.9 | same | 2.3 |
| 2.0 | 1.1 | 1.2 | 0.9 | same | 3.5 |
| **4.0** | **4.0** | **0.4** | **0.2** | switches to SAFE paths | **5.6** |
| 8.0 | 4.0 | 0.4 | 0.2 | same | 7.2 |
| 16.0 | 4.0 | 0.4 | 0.2 | same | 10.4 |

👉 Παρατηρείται **φάση μεταγωγής (phase transition)** κοντά στο  
**λ ≈ 4 → planner προτιμά ασφαλείς στόχους αντί “γρήγορους αλλά επικίνδυνους”**

---

## 5️⃣ Τι σημαίνουν αυτά για paper
Αυτά τα αποτελέσματα υποστηρίζουν:

### ✔ Clearly defined Safety–Performance Trade-off
- threshold τ → *πότε ενεργοποιείται ασφάλεια*
- λ → *πόσο “βαριά” τιμωρείται το ρίσκο*

### ✔ Καθαρή πειραματική απόδειξη
- SAFE mode μειώνει risk metrics
- αυξάνει κόστος
- αλλά δίνει predictable συμπεριφορά

### ✔ Policies μπορούν να βελτιωθούν με:
- dynamic τ
- adaptive λ
- per-robot trust

---

## ✔️ Περιλαμβανόμενα Scripts
| Script | Περιγραφή |
|--------|-----------|
| `multi_robot_safe_mode_experiment.py` | Βασική αυτονομία + safe switching |
| `multi_robot_safe_mode_logged.py` | Logging αποτελεσμάτων |
| `safe_mode_threshold_ablation.py` | Threshold experiment |
| `analyze_lambda_sweep_risk_length.py` | Risk vs Length λ-sweep |

Όλα τα αποτελέσματα αποθηκεύονται σε CSV και μπορούν να χρησιμοποιηθούν σε plots & paper figures.

---

#Το σύστημα:
- αναγνωρίζει πότε “δεν εμπιστεύεται τον εαυτό του”
- ενεργοποιεί ασφαλέστερη πολιτική
- μειώνει ρίσκο
- παρέχει καθαρό θεωρητικό trade-off
- και είναι reproducible με clean experiments



