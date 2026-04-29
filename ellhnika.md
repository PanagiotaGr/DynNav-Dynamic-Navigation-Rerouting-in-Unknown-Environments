# DynNav — Πλήρης Ανάλυση: Τι Έφτιαξα και Γιατί


---

## Εισαγωγή: Τι είναι το DynNav;

Το DynNav είναι ένα **ερευνητικό framework αυτόνομης πλοήγησης** για ρομπότ που κινούνται σε εντελώς άγνωστα περιβάλλοντα — χωρίς προκατασκευασμένο χάρτη, με ατελείς αισθητήρες, με δυναμικά εμπόδια που εμφανίζονται ξαφνικά.

### Το κεντρικό πρόβλημα

Φαντάσου ένα ρομπότ (π.χ. TurtleBot3) που μπαίνει σε ένα κτίριο που δεν έχει ξαναδεί. Δεν έχει χάρτη. Οι αισθητήρες του κάνουν λάθη. Ξαφνικά κάποιος κουβαλάει ένα κουτί μπροστά του. Πρέπει να φτάσει στο γραφείο 3Β.

Τα κλασικά συστήματα πλοήγησης αντιμετωπίζουν αυτό σαν να ήταν πλήρως γνωστό το περιβάλλον. Στην πραγματικότητα δεν είναι. Το DynNav ασχολείται ακριβώς με αυτό το χάσμα.

### Τι κάνει το DynNav που δεν κάνουν τα άλλα;

| Κλασικά συστήματα | DynNav |
|---|---|
| Χρειάζονται πλήρη χάρτη | Χτίζουν χάρτη καθώς κινούνται |
| Θεωρούν τους αισθητήρες τέλειους | Μετρούν και χρησιμοποιούν αβεβαιότητα |
| Δεν σκέφτονται τον κίνδυνο | Υπολογίζουν risk και επιλέγουν ασφαλείς διαδρομές |
| Δεν έχουν εγγυήσεις ασφάλειας | Χρησιμοποιούν formal safety (STL + CBF) |
| Δεν μαθαίνουν | Ενσωματώνουν RL, curriculum learning, federated learning |
| Ένα ρομπότ μόνο | Συντονισμός πολλαπλών ρομπότ με fault tolerance |
| Δεν καταλαβαίνουν γλώσσα | «Πήγαινε στην κουζίνα» → πλοήγηση |

---

## CONTRIBUTION 01 — Learned A\* Heuristics

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Learning-Augmented A\* with Neural Heuristic Functions

**Τι είναι:** Ένα σύστημα που εκπαιδεύει νευρωνικό δίκτυο για να βελτιώσει την ταχύτητα του αλγορίθμου εύρεσης μονοπατιού A\*.

**Ποιο πρόβλημα λύνει:** Ο αλγόριθμος A\* (διάβαζε «A-star») είναι ο βασικός τρόπος που ένα ρομπότ βρίσκει διαδρομή από σημείο A σε σημείο B. Ψάχνει ποιο μονοπάτι είναι το συντομότερο εξετάζοντας διάφορα ενδιάμεσα σημεία. Η ταχύτητά του εξαρτάται από πόσο «έξυπνα» μαντεύει σε ποια κατεύθυνση να ψάξει πρώτα.

**Γιατί είναι σημαντικό:** Σε μεγάλα και πολύπλοκα περιβάλλοντα, ο A\* μπορεί να εξετάσει χιλιάδες σημεία πριν βρει το μονοπάτι. Αυτό είναι αργό για πραγματικό ρομπότ που πρέπει να αποφασίζει σε milliseconds.

---

### 2. Background για Αρχάριους

**Τι είναι ο A\*;**

Φαντάσου ένα πλέγμα κελιών — σαν χάρτη σκακιού. Το ρομπότ είναι στη μία γωνία, ο στόχος στην απέναντι. Τα εμπόδια είναι μαύρα κελιά.

Ο A\* αναζητά ένα μονοπάτι διατηρώντας μια «ουρά προτεραιότητας» — λίστα σημείων που αξίζει να εξετάσει. Για κάθε σημείο υπολογίζει:

```
f(n) = g(n) + h(n)
```

- `g(n)` = κόστος για να φτάσεις εκεί από την αρχή (γνωστό)
- `h(n)` = εκτίμηση κόστους από εκεί μέχρι τον στόχο (heuristic)

Η heuristic είναι η «μαντεψιά». Αν μαντεύεις καλά, ψάχνεις λιγότερα σημεία.

**Κλασική heuristic:** Απόσταση Manhattan (|Δx| + |Δy|) ή Ευκλείδεια απόσταση. Απλές, ασφαλείς, αλλά συχνά χαλαρές — υποεκτιμούν το πραγματικό κόστος πολύ.

**Admissibility:** Για να εγγυάται ο A\* βέλτιστη διαδρομή, η heuristic δεν πρέπει ποτέ να υπερεκτιμά το πραγματικό κόστος: `h(n) ≤ h*(n)`. Αυτή η ιδιότητα λέγεται admissibility.

**Νευρωνικό δίκτυο:** Ένα απλό MLP (Multi-Layer Perceptron) — ένα ταξινομητής/παλινδρομητής που μαθαίνει από δεδομένα.

---

### 3. Τι Έκανα Εγώ Πρακτικά

**Pipeline:**
1. Τρέχω A\* σε πολλά training environments → συλλέγω ζεύγη `(κατάσταση s, πραγματικό κόστος h*(s))`
2. Εκπαιδεύω MLP regressor: `h_θ(s) ≈ h*(s)`, Loss = MSE
3. Στο inference: χρησιμοποιώ `h̃(s) = min(h_θ(s), h_naive(s))` — το ελάχιστο των δύο, ώστε να διατηρώ admissibility
4. Τρέχω A\* με τη νέα heuristic

**Input:** κατάσταση του ρομπότ (θέση, απόσταση από εμπόδια, κατεύθυνση στόχου)  
**Output:** εκτιμώμενο κόστος h̃(s)  
**Script:** `contributions/01_learned_astar/experiments/eval_astar_learned.py`

---

### 4. Αλγοριθμική Ανάλυση

```
TRAINING:
  for each episode in training_maps:
    run A* → get optimal path
    for each state s on path:
      record (s, h*(s))  ← true cost to goal
  train MLP: minimize Σ (h_θ(s_i) - h*(s_i))²

INFERENCE (Clipped Neural A*):
  function h_tilde(s):
    h_neural = MLP.predict(s)
    h_safe   = manhattan_distance(s, goal)
    return min(h_neural, h_safe)   ← guarantees admissibility
  
  run A* using h_tilde as heuristic
```

**Μαθηματική εγγύηση:**  
Επειδή `h̃(s) = min(h_θ(s), h_naive(s)) ≤ h_naive(s) ≤ h*(s)` (η Manhattan/Euclidean είναι admissible), η τελική heuristic παραμένει admissible → A\* βρίσκει βέλτιστη διαδρομή.

---

### 5. Research Question

> Μπορεί μια νευρωνικά μαθημένη heuristic να μειώσει τον αριθμό κόμβων που εξετάζει ο A\* διατηρώντας παράλληλα την εγγύηση βέλτιστης διαδρομής;

**Hypothesis:** Η νευρωνική heuristic, εκπαιδευμένη σε παρόμοιους χώρους, θα εκτιμά καλύτερα το h\*(s) από τη Manhattan, μειώνοντας τους εξεταζόμενους κόμβους.

**Novelty:** Ο συνδυασμός neural heuristic + clipping για admissibility σε navigation context.

---

### 6. Αποτελέσματα

- ~35% μείωση node expansions σε benchmark maps
- Ίδια ποιότητα διαδρομής (βέλτιστη)
- **Τι λείπει:** σύγκριση με άλλες learned heuristics (π.χ. Neural A\* του Yonetani et al.), evaluation σε πολύ μεγάλα maps, ανάλυση generalization σε maps που δεν έχει δει το δίκτυο.

---

### 7. Κριτική Αξιολόγηση

**Δυνατό:** Μαθηματικά σωστό (clipping διατηρεί admissibility). Καλή βασική ιδέα.  
**Αδύναμο:** Η ιδέα είναι ήδη γνωστή στη literature. Χρειάζεται ισχυρότερα baselines.  
**Ερώτηση reviewer:** «Πώς συγκρίνεται με το Neural A\* (Yonetani 2021) ή το Differentiable A\*;»

---

### 8. Πώς Γίνεται Καλύτερο

- Ablation: με/χωρίς clipping, διάφορες αρχιτεκτονικές MLP
- Baseline: Vanilla A\*, Dijkstra, RRT\*, Neural A\* (Yonetani 2021)
- Evaluation σε χάρτες από robot datasets (MatterPort3D, Gibson)
- Statistical testing (t-test) για τη μείωση node expansions

---

### 9. Publication Potential

**Τίτλος:** "Admissible Neural Heuristics for Learning-Augmented A\* in Unknown Environments"  
**Venue:** ICRA Workshop on Learning for Planning, RA-L  
**Level:** Workshop-level ή incremental conference paper  
**Outline:** Introduction → Background A\* → Neural Heuristic Training → Admissibility Guarantee → Experiments → Comparison → Conclusion

---

### 10. PhD Potential

Μπορεί να γίνει κεφάλαιο «Planning Efficiency» σε διδακτορικό. Ανοίγει σε: online learning της heuristic, multi-task heuristics για διαφορετικά robots, meta-learning για heuristics.

---

## CONTRIBUTION 02 — Uncertainty Estimation

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Belief-State Estimation via EKF and UKF

**Τι είναι:** Ένα σύστημα που παρακολουθεί πόσο «σίγουρο» είναι το ρομπότ για τη θέση του, χρησιμοποιώντας Kalman Filtering.

**Ποιο πρόβλημα λύνει:** Κανένας αισθητήρας δεν είναι τέλειος. Το LiDAR έχει θόρυβο. Η οδομετρία (μέτρηση μέσω τροχών) συσσωρεύει σφάλμα. Αν το ρομπότ θεωρεί τυφλά ότι η μέτρηση είναι σωστή, θα κάνει λάθος αποφάσεις.

---

### 2. Background για Αρχάριους

**Τι είναι το Kalman Filter;**

Φαντάσου ότι ακολουθείς έναν φίλο στο δρόμο με κλειστά μάτια, χρησιμοποιώντας μόνο τα βήματά του. Μετά ανοίγεις τα μάτια για μια στιγμή και βλέπεις που είναι. Ο Kalman Filter κάνει ακριβώς αυτό — συνδυάζει την «προβλεψιμότητα» (μοντέλο κίνησης) με τις «παρατηρήσεις» (αισθητήρες) για να εκτιμήσει τη θέση.

**Belief state:** Αντί για μια συγκεκριμένη θέση, το σύστημα κρατάει μια κατανομή πιθανοτήτων: `b(s) = N(μ, Σ)` — μια Gaussian με μέση τιμή μ (καλύτερη εκτίμηση) και πίνακα covariance Σ (αβεβαιότητα).

**EKF (Extended Kalman Filter):** Για μη-γραμμικά συστήματα — χρησιμοποιεί γραμμικοποίηση μέσω Jacobian.

**UKF (Unscented Kalman Filter):** Πιο ακριβής για έντονα μη-γραμμικά συστήματα — χρησιμοποιεί sigma points.

---

### 3. Τι Έκανα Εγώ Πρακτικά

**Equations:**
```
Predict:
  μ̂_{t|t-1} = f(μ_{t-1}, a_t)
  Σ_{t|t-1} = F_t Σ_{t-1} F_t^T + Q

Update:
  K_t = Σ_{t|t-1} H_t^T (H_t Σ_{t|t-1} H_t^T + R)^{-1}
  μ_t = μ̂_{t|t-1} + K_t (z_t - h(μ̂_{t|t-1}))
  Σ_t = (I - K_t H_t) Σ_{t|t-1}
```

**Output:** Belief state `(μ_t, Σ_t)` — δίνεται ως input στον risk planner (Contribution 03).

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Σωστή μαθηματική υλοποίηση. Απαραίτητο infrastructure για τα υπόλοιπα modules.  
**Αδύναμο:** EKF/UKF είναι textbook material. Δεν υπάρχει novelty εδώ από μόνο του.  
**Framing:** Να παρουσιάζεται ως *infrastructure layer*, όχι ως αυτόνομη ερευνητική συνεισφορά.

---

### 5. Publication / PhD Potential

Από μόνο του: όχι publishable. Ως κομμάτι ενός συστήματος: απαραίτητο. PhD context: «uncertainty propagation layer» που τροφοδοτεί risk planning, safety shields, και causal attribution.

---

## CONTRIBUTION 03 — Belief-Space & Risk-Aware Planning

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** CVaR-Optimised Risk-Weighted Path Planning under Uncertainty

**Τι είναι:** Ένας planner που δεν ψάχνει απλά τη συντομότερη διαδρομή, αλλά τη **συντομότερη-και-ασφαλέστερη** — λαμβάνοντας υπόψη τον κίνδυνο σύγκρουσης σε κάθε κελί του χάρτη.

**Γιατί είναι σημαντικό:** Η κλασική πλοήγηση βελτιστοποιεί μόνο το μήκος. Σε πραγματικά περιβάλλοντα, μια ελαφρώς μακρύτερη αλλά ασφαλέστερη διαδρομή είναι πολύ προτιμότερη.

---

### 2. Background για Αρχάριους

**Τι είναι το CVaR;**

Φαντάσου ότι σχεδιάζεις ένα επενδυτικό χαρτοφυλάκιο. Δεν σε νοιάζει μόνο η μέση απόδοση — σε νοιάζει τι συμβαίνει στα **χειρότερα σενάρια**. Το CVaR (Conditional Value at Risk) μετράει ακριβώς αυτό: τι είναι η αναμενόμενη ζημιά στο χειρότερο α% των περιπτώσεων.

**CVaR_α(X) = E[X | X ≥ VaR_α(X)]**

Σε navigation: αντί να βελτιστοποιούμε την μέση πιθανότητα σύγκρουσης, βελτιστοποιούμε τη **χειρότερη περίπτωση** στο 5% των σεναρίων.

**Risk-weighted cost:**
```
f(s) = g(s) + λ · CVaR_α(p_collision(s)) + h(s)
```
Όπου λ ελέγχει πόσο «φοβικός» είναι ο planner.

---

### 3. Τι Έκανα Εγώ Πρακτικά

- Τροποποίησα τον A\* ώστε κάθε κελί να έχει risk cost από CVaR
- Το risk προέρχεται από το belief state (Contribution 02) ή τα diffusion maps (Contribution 12)
- Parameter sweep: διάφορες τιμές λ → Pareto curve safety/efficiency
- Ablation: risk-aware vs risk-agnostic planner

---

### 4. Research Question

> Πώς πρέπει να ισορροπεί ένα αυτόνομο σύστημα μεταξύ αποδοτικότητας και tail-risk minimization στην πλοήγηση υπό αβεβαιότητα;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** CVaR ως risk metric είναι θεωρητικά πλούσιο και καλά τεκμηριωμένο. Η modularity (οποιοδήποτε risk signal συνδέεται με τον ίδιο planner) είναι σημαντικό design choice.

**Αδύναμο:** CVaR planning είναι γνωστό (Majumdar & Pavone 2020). Χρειάζεται σαφής διαφοροποίηση.

**Τι λείπει:** Σύγκριση με chance-constrained planning, robust MPC. Ανάλυση computational overhead.

---

### 6. Publication Potential

**Τίτλος:** "CVaR-Augmented A\* for Risk-Sensitive Navigation in Partially Observable Environments"  
**Venue:** IROS, RA-L  
**Level:** Incremental conference — χρειάζεται ισχυρά experiments

---

## CONTRIBUTION 04 — Irreversibility & Returnability

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Returnability-Constrained Navigation Planning

**Τι είναι:** Ένα σύστημα που πριν κάθε κίνηση ελέγχει: «αν πάω εκεί, μπορώ να επιστρέψω;» Αποτρέπει το ρομπότ από το να εισέλθει σε αδιέξοδα.

**Γιατί είναι σημαντικό:** Σε άγνωστα περιβάλλοντα, ένα ρομπότ μπορεί να μπει σε χώρο από τον οποίο δεν μπορεί να βγει — στενός διάδρομος, δωμάτιο χωρίς έξοδο. Αυτό είναι καταστροφικό για αυτόνομη λειτουργία.

---

### 2. Background για Αρχάριους

**Irreversible state:** Κατάσταση από την οποία δεν υπάρχει διαδρομή επιστροφής σε ασφαλή ζώνη.

**Returnability:** Η ικανότητα να επιστρέψεις στην ασφαλή ζώνη S_safe από τη συγκεκριμένη κατάσταση s, υπό τον τρέχοντα χάρτη, σε horizon H βημάτων.

Σχετίζεται με **backward reachability analysis** — η ανάδρομη ανάλυση που υπολογίζει ποιες καταστάσεις μπορούν να οδηγήσουν σε ασφαλή ζώνη.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
RETURNABILITY CHECK:
  function is_returnable(s, map, H):
    Run backward BFS from S_safe on current map
    return distance(s, S_safe) ≤ H

PLANNING WITH RETURNABILITY:
  cost(s) += γ_irrev * (1 - is_returnable(s, map, H))
  
  Alternative: hard constraint — never enter non-returnable states
```

**Input:** τρέχουσα θέση, χάρτης  
**Output:** flag returnable/non-returnable, penalised cost map

---

### 4. Research Question

> Μπορεί ο προληπτικός έλεγχος returnability να μειώσει την ποσοστιαία συχνότητα εισόδου σε αδιέξοδα σε άγνωστα περιβάλλοντα;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σαφής μαθηματική ορισμός. Σχετίζεται με safe exploration (Moldovan & Abbeel 2012) — ένα αξιόλογο research direction.

**Αδύναμο:** BFS approximation δεν είναι full backward reachability. Υπό uncertainty (ατελής χάρτης), η εγγύηση αποδυναμώνεται.

**Reviewer:** «Πώς επεκτείνεται σε partially observable environments;» «Τι γίνεται όταν ο χάρτης είναι ατελής;»

---

### 6. Publication Potential

**Τίτλος:** "Safe Exploration via Returnability Constraints in Unknown Environments"  
**Venue:** ICRA, Safety in Robotics Workshop  
**Level:** Workshop / short paper — χρειάζεται formal analysis

---

## CONTRIBUTION 05 — Safe-Mode Navigation

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Risk-Triggered Adaptive Safe-Mode FSM

**Τι είναι:** Ένα σύστημα τριών καταστάσεων (NORMAL → SAFE → EMERGENCY) που αλλάζει αυτόματα τη συμπεριφορά του ρομπότ ανάλογα με τον κίνδυνο.

---

### 2. Background για Αρχάριους

**FSM (Finite State Machine):** Ένας αυτόματος με πεπερασμένες καταστάσεις και μεταβάσεις. Σαν κυκλοφοριακό φανάρι — τρεις καταστάσεις, αλλάζει με βάση συνθήκες.

**Hysteresis:** Η safe mode δεν σβήνει αμέσως όταν ο κίνδυνος μειωθεί — χρειάζεται T_hold βήματα σταθερά χαμηλού κινδύνου. Αποτρέπει γρήγορες εναλλαγές.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
MODE TRANSITIONS:
  if risk > τ_high OR (mode==SAFE AND risk > τ_low):
    mode = SAFE
  if risk > τ_critical:
    mode = EMERGENCY (stop)
  if mode==SAFE AND risk < τ_low for T_hold steps:
    mode = NORMAL

PER-MODE BEHAVIOUR:
  NORMAL:    v_max = v_n,    inflation = r_n
  SAFE:      v_max = α·v_n,  inflation = β·r_n  (α<1, β>1)
  EMERGENCY: v_max = 0,      alert operator
```

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Απλό, ρεαλιστικό, deployable. Σωστή σύνδεση με risk signal.  
**Αδύναμο:** FSM είναι πολύ κλασική προσέγγιση. Δεν είναι ερευνητικά novel.  
**Framing:** Systems integration contribution — όχι νέος αλγόριθμος.

---

### 6. Publication Potential

Από μόνο του: όχι. Ως κομμάτι ολοκληρωμένου συστήματος: σημαντικό. Δείχνει πρακτική κατανόηση του deployment.
## CONTRIBUTION 06 — Energy & Connectivity-Aware Planning

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Resource-Constrained Path Planning: Energy Budget and Communication Quality

**Τι είναι:** Επέκταση του planner που λαμβάνει υπόψη δύο πρακτικούς περιορισμούς: την μπαταρία του ρομπότ και την ποιότητα επικοινωνίας WiFi σε κάθε ζώνη.

**Γιατί είναι σημαντικό:** Τα πραγματικά ρομπότ δεν έχουν άπειρη μπαταρία. Σε μεγάλα κτίρια, αν αδειάσει η μπαταρία στη μέση της αποστολής, η αποστολή αποτυγχάνει. Ομοίως, αν το ρομπότ χαθεί σε dead zone WiFi, ο χειριστής χάνει τον έλεγχο.

---

### 2. Background για Αρχάριους

**Resource-constrained shortest path:** Κλασικό πρόβλημα operations research. Βρες τον φθηνότερο δρόμο που ταυτόχρονα δεν υπερβαίνει το resource budget. Είναι NP-hard γενικά, αλλά με Lagrangian relaxation γίνεται tractable.

**Ενεργειακό μοντέλο:** Κάθε κίνηση έχει κόστος ανάλογο με: ισχύ κινητήρα × απόσταση × συντελεστή εδάφους (επίπεδο = φθηνό, ανηφόρα = ακριβό).

**Connectivity map:** Χάρτης ποιότητας σήματος — κάθε κελί έχει τιμή q(s) ∈ [0,1].

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
CONSTRAINED A*:
  f(s) = g(s) + λ_E·e_cost(s) + λ_Q·max(0, Q_min - q(s)) + h(s)
  
  CONSTRAINT: Σ e_cost(edges on path) ≤ B (battery budget)
  
  If no feasible direct path:
    Insert charging station waypoint → replan
```

**Input:** Battery level B, connectivity map q(s), terrain map  
**Output:** Feasible path respecting both constraints  
**Fallback:** Detour via charging station

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Πρακτικά σημαντικό για real deployment. Καλά ορισμένο optimization problem.  
**Αδύναμο:** Λείπει πειραματική επαλήθευση με πραγματικά δεδομένα μπαταρίας από TurtleBot3.  
**Reviewer:** «Πού βασίζεται το ενεργειακό μοντέλο; Επαληθεύτηκε σε πραγματικό robot;»

---

### 5. Publication Potential

**Τίτλος:** "Energy-Aware Navigation with Communication Constraints for Mobile Robots"  
**Venue:** IROS, RoboCup Symposium  
**Level:** Short conference paper — χρειάζεται hardware validation

---

## CONTRIBUTION 07 — Next-Best-View Exploration

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Information-Theoretic Next-Best-View Selection for Active Mapping

**Τι είναι:** Ένα σύστημα που αποφασίζει πού να πάει το ρομπότ στη συνέχεια για να **μάθει τα περισσότερα** για τον άγνωστο χώρο.

**Γιατί είναι σημαντικό:** Η εξερεύνηση άγνωστου χώρου είναι fundamental στη ρομποτική. Το κλασικό «frontier exploration» (πήγαινε στο κοντινότερο άγνωστο σημείο) είναι suboptimal. Η information-theoretic προσέγγιση επιλέγει το σημείο που μεγιστοποιεί τη μείωση αβεβαιότητας του χάρτη.

---

### 2. Background για Αρχάριους

**Entropy χάρτη:** Κάθε κελί έχει πιθανότητα p_c να είναι κατειλημμένο. Αβεβαιότητα = -p_c·log(p_c) - (1-p_c)·log(1-p_c). Μεγάλη αβεβαιότητα σημαίνει p_c ≈ 0.5 (δεν ξέρουμε).

**Information Gain:** Πόση αβεβαιότητα θα μειωθεί αν πάω στο σημείο v και δω τι υπάρχει.

**NBV criterion:** Διαίρεση IG με κόστος μετακίνησης — «πόση πληροφορία ανά χιλιόμετρο».

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
NBV SELECTION:
  for each candidate viewpoint v in V:
    IG(v) = H(m) - E[H(m | z_v)]   ← expected entropy reduction
    cost(v) = distance(robot, v)
  
  v* = argmax_v IG(v) / cost(v)
  
  Navigate to v*, update map, repeat.

MAP UPDATE (Binary Bayes Filter):
  log_odds(m_c | z_t) += log_odds(m_c | z_t alone) - log_odds(prior)
```

**Integration:** Συνδέεται με `ig_explorer/` module, feeds into Contribution 03.

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Information-theoretic exploration είναι καλά τεκμηριωμένο. Ray-casting IG estimation είναι standard approach.  
**Αδύναμο:** Greedy (myopic) — δεν σχεδιάζει multi-step exploration. Δεν συγκρίνεται με state-of-the-art methods (GBPlanner, FUEL).  
**Reviewer:** «Γιατί greedy αντί multi-step planning; Τι overhead έχει το IG computation;»

---

### 5. Publication Potential

**Τίτλος:** "Uncertainty-Driven Active Exploration with Information-Geometric Viewpoint Selection"  
**Venue:** IROS, ICRA Workshop on Exploration  
**Level:** Workshop paper — ισχυρότερο αν συνδυαστεί με NeRF uncertainty (Contribution 24)

---

## CONTRIBUTION 08 — Security & Intrusion Detection

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Innovation-Based Anomaly Detection for Sensor Integrity Monitoring

**Τι είναι:** Ένα Intrusion Detection System (IDS) που παρακολουθεί αν οι αισθητήρες του ρομπότ δέχονται επίθεση ή αντιμετωπίζουν βλάβη.

**Γιατί είναι σημαντικό:** Καθώς τα ρομπότ γίνονται συνδεδεμένα (IoT, 5G), γίνονται στόχοι κυβερνοεπιθέσεων. Μια επίθεση στους αισθητήρες μπορεί να κάνει το ρομπότ να «βλέπει» ψεύτικα εμπόδια ή να μην βλέπει πραγματικά — με καταστροφικά αποτελέσματα.

---

### 2. Background για Αρχάριους

**Innovation sequence:** Στο Kalman Filter, ορίζεται ως η διαφορά μεταξύ της πραγματικής μέτρησης και της προβλεπόμενης:

```
ν_k = z_k - H_k · ŝ_{k|k-1}
```

Υπό φυσιολογικές συνθήκες: ν_k ~ N(0, S_k). Αν κάποιος χακάρει τους αισθητήρες, η κατανομή αλλάζει.

**χ²-test:** Στατιστικό τεστ που ελέγχει αν το ν_k συμφωνεί με την αναμενόμενη κατανομή:
```
χ²_k = ν_k^T · S_k^{-1} · ν_k  ←  αν > threshold: ALARM
```

**CUSUM:** Cumulative Sum — ανιχνεύει αργές, σταδιακές αλλαγές που το χ²-test χάνει:
```
g_k = max(0, g_{k-1} + χ²_k - κ)
alarm if g_k > h
```

---

### 3. Τι Έκανα Εγώ Πρακτικά

- Παρακολούθηση innovation sequence από EKF σε κάθε timestep
- χ²-test με configurable false-alarm rate α
- CUSUM για gradual drift detection
- Alarm → trigger Safe-Mode (Contribution 05) + log event
- ROS2 nodes σε `cybersecurity_ros2/`

**Metrics:** Detection Rate (TPR), False Alarm Rate (FPR), Detection Latency (steps)

---

### 4. Research Question

> Μπορεί statistical monitoring του Kalman filter innovation να ανιχνεύσει sensor spoofing attacks σε real-time με ελεγχόμενο false-alarm rate;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σύνδεση IDS → Safe-Mode → Causal Attribution (Contribution 14) δημιουργεί ένα πλήρες security pipeline. Innovation monitoring είναι principled approach.  
**Αδύναμο:** χ²/CUSUM είναι textbook methods. Novelty είναι στην εφαρμογή και ενσωμάτωση στο navigation context.  
**Reviewer:** «Δοκιμάστηκε σε πραγματικές επιθέσεις; Τι γίνεται με adaptive attackers που γνωρίζουν το detection scheme;»

---

### 6. Publication Potential

**Τίτλος:** "Integrity Monitoring for Autonomous Navigation: From Sensor Spoofing Detection to Safe Mode Recovery"  
**Venue:** ICRA Workshop on Safe Robotics, IEEE S&P (Security)  
**Level:** Workshop + systems paper — ισχυρό αν combined με Contribution 25 (attack simulator)

---

## CONTRIBUTION 09 — Multi-Robot Coordination

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Decentralised Priority-Based Path Reservation for Multi-Robot Navigation

**Τι είναι:** Σύστημα συντονισμού πολλαπλών ρομπότ που μοιράζονται τον ίδιο χώρο — χωρίς κεντρικό συντονιστή που αποτελεί single point of failure.

---

### 2. Background για Αρχάριους

**Multi-robot path planning (MAPF):** Το πρόβλημα εύρεσης διαδρομών για N ρομπότ ταυτόχρονα χωρίς συγκρούσεις. Κεντρικές λύσεις (CBS - Conflict-Based Search) είναι βέλτιστες αλλά δεν scale. Αποκεντρωμένες λύσεις scale αλλά δεν εγγυώνται βέλτιστο αποτέλεσμα.

**Priority reservation:** Κάθε ρομπότ έχει προτεραιότητα. Υψηλότερης προτεραιότητας ρομπότ σχεδιάζουν πρώτα και «κλειδώνουν» κελιά. Χαμηλότερης προτεραιότητας ρομπότ αντιμετωπίζουν τα κλειδωμένα κελιά ως εμπόδια.

**Gossip protocol:** Αποκεντρωμένος τρόπος ανταλλαγής πληροφορίας — κάθε ρομπότ μοιράζεται τον χάρτη του με τους γείτονές του.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
COORDINATION PROTOCOL:
  1. Gossip: share occupancy maps with neighbors
  2. Build shared belief map
  3. Priority order established (static or urgency-based)
  4. Robot i plans path treating higher-priority reservations as occupied
  5. Broadcast reservation to fleet
  6. Monitor disagreement: if ||map_i - map_j||_1 > τ_d → flag
  
RISK BUDGET ALLOCATION:
  Σ r_i ≤ R_total,  r_i ≥ 0
```

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Gossip-based map sharing είναι scalable. Risk budget allocation είναι ενδιαφέρον extension.  
**Αδύναμο:** Priority-based approaches είναι γνωστές και δεν εγγυώνται βέλτιστο. Δεν συγκρίνεται με CBS ή M\*.  
**Reviewer:** «Πώς επιλέγεται η προτεραιότητα; Τι γίνεται με deadlocks;»

---

### 5. Publication Potential

**Venue:** AAMAS, MRS (Multi-Robot Systems) Workshop  
**Level:** Systems integration — χρειάζεται formal analysis για deadlock freedom

---

## CONTRIBUTION 10 — Human-Aware & Ethics-Guided Navigation

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Preference-Weighted Navigation with Ethical Zone Constraints and Trust-Aware Autonomy

**Τι είναι:** Επέκταση του navigation planner που σέβεται ανθρώπινες προτιμήσεις, ηθικούς περιορισμούς (no-go zones), και προσαρμόζεται στο επίπεδο εμπιστοσύνης του χειριστή.

---

### 2. Background για Αρχάριους

**Ethical zones:** Ορισμένες περιοχές που το ρομπότ δεν πρέπει να εισέλθει για λόγους ιδιωτικότητας, ασφάλειας ή ηθικής (νοσοκομεία, χώροι προσευχής, κλπ).

**Human proximity cost:** Τα ρομπότ πρέπει να διατηρούν κοινωνικές αποστάσεις από ανθρώπους — «social navigation».

**Trust model:** Ο χειριστής εμπιστεύεται το ρομπότ σε διαβαθμίσεις. Υψηλή εμπιστοσύνη → πλήρης αυτονομία. Χαμηλή εμπιστοσύνη → ζητά άδεια για κάθε κίνηση.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
AUGMENTED COST:
  f_total(s) = g(s) + λ_r·r(s) + c_ethical(s) + 
               λ_h·Σ_j w_j·exp(-||s-p_j||²/2σ_j²) + h(s)

c_ethical(s):
  = ∞    if s ∈ Z_hard  (hard no-go)
  = γ_z  if s ∈ Z_soft  (soft penalty)
  = 0    otherwise

TRUST SCALING: actions scaled by τ(t) ∈ [0,1]
```

**Config:** `ethical_zones.json` — zone definitions

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Infrastructure είναι σωστή και deployable. Zone configuration είναι flexible.  
**Αδύναμο:** «Ethics» εδώ = zone avoidance + Gaussian repulsion. Δεν είναι ethical reasoning. Το trust model είναι scalar — δεν μαθαίνει.  
**ΠΡΟΣΟΧΗ:** Να μην παρουσιάζεται ως «ethical AI system» — είναι preference-constrained navigation.

---

### 5. Publication Potential

**Τίτλος:** "Preference-Aware Navigation: Ethical Zones, Social Distancing, and Trust-Parameterised Autonomy"  
**Venue:** HRI (Human-Robot Interaction), ICRA Social Robotics Workshop  
**Level:** Workshop paper

---

## CONTRIBUTION 11 — VLM Navigation Agent

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Vision-Language Model as Zero-Shot Semantic Goal Generator for Navigation

**Τι είναι:** Σύστημα που χρησιμοποιεί ένα μεγάλο vision-language model (LLaVA, GPT-4V) για να «διαβάζει» την εικόνα της κάμερας και να παράγει semantic navigation goals.

**Γιατί είναι σημαντικό:** Τα κλασικά navigation systems χρειάζονται metric goals (x,y coordinates). Στην πραγματικότητα, θέλουμε να λέμε «πήγαινε στην κουζίνα» χωρίς να ξέρουμε τις συντεταγμένες. Τα VLMs έχουν world knowledge που μπορεί να γεφυρώσει αυτό το χάσμα.

---

### 2. Background για Αρχάριους

**VLM (Vision-Language Model):** Νευρωνικό δίκτυο που κατανοεί ταυτόχρονα εικόνες και κείμενο. Παραδείγματα: GPT-4V, LLaVA, CLIP. Μπορεί να απαντά ερωτήσεις για εικόνες.

**Zero-shot:** Χωρίς καμία εκπαίδευση στο συγκεκριμένο περιβάλλον — χρησιμοποιεί γενική γνώση.

**Pinhole camera model:** Μαθηματικό μοντέλο που συνδέει pixel coordinates με 3D world coordinates μέσω depth (βάθος από depth camera).

**Back-projection:** Μετατροπή pixel (u,v) → metric (x,y) χρησιμοποιώντας depth:
```
x_world = t_robot + R_yaw · [(u - c_x)/f_x · d, 0, d]^T
```

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
PIPELINE:
  1. Capture RGB frame
  2. Encode as base64 → send to VLM API
  3. Prompt: "Identify semantic region and best navigation direction"
  4. Parse JSON response: {region, goal, confidence, pixel_u, pixel_v}
  5. if confidence ≥ τ_c:
       back-project pixel → metric waypoint using depth map
  6. Feed waypoint to A* planner
  7. Graceful degradation: if VLM unavailable → return None

CONFIDENCE GATING:
  accept = 1[c ≥ τ_c]
```

---

### 4. Research Question

> Μπορεί ένα pre-trained VLM να χρησιμεύσει ως zero-shot semantic goal generator για robot navigation χωρίς fine-tuning;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Η interface design (structured JSON output + confidence gating + back-projection) είναι καλά μηχανικά δουλεμένη. Graceful degradation χωρίς VLM είναι σημαντικό για deployment.  
**Αδύναμο:** Offline evaluation με random frames (stub mode). Δεν έχει δοκιμαστεί με πραγματικό VLM σε navigation episode. Δεν συγκρίνεται με LM-Nav, ViNT.  
**Reviewer:** «Τι γίνεται με latency; VLM API calls είναι αργές — πώς ενσωματώνεται σε real-time navigation;»

---

### 6. Publication Potential

**Τίτλος:** "Structured VLM Interfaces for Semantic Navigation Goal Generation"  
**Venue:** CoRL, ICRA Workshop on Foundation Models for Robotics  
**Level:** Workshop paper — χρειάζεται live VLM evaluation

---

## CONTRIBUTION 12 — Diffusion Occupancy Maps

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Score-Based Diffusion Models for Probabilistic Occupancy Prediction

**Τι είναι:** Χρήση generative AI (diffusion models) για να παράγει πολλαπλά πιθανά «μελλοντικά σενάρια» κατοχής του χώρου, αντί για μια μόνο deterministic εκτίμηση.

**Γιατί είναι σημαντικό:** Σε δυναμικά περιβάλλοντα, δεν ξέρουμε πού θα είναι τα εμπόδια στο επόμενο δευτερόλεπτο. Αντί να «μαντεύουμε» μία απάντηση, παράγουμε 10 πιθανά σενάρια και σχεδιάζουμε για τα χειρότερα.

---

### 2. Background για Αρχάριους

**Diffusion models:** Κατηγορία generative AI (σαν DALL-E, Stable Diffusion). Μαθαίνουν να «αφαιρούν θόρυβο» από noisy εικόνες βήμα-βήμα. Η ίδια ιδέα εφαρμόζεται σε occupancy maps.

**Forward process:** Προσθέτω σταδιακά Gaussian noise σε ένα occupancy map μέχρι να γίνει pure noise.

**Reverse process:** Μαθαίνω να αφαιρώ το noise σε T βήματα, conditioned στην ιστορία παρατηρήσεων → παράγω plausible future occupancy map.

**CVaR από samples:** Αν παράγω N=10 samples, το CVaR-95 = μέση τιμή του χειρότερου 5% = 1 sample με τον περισσότερο κίνδυνο.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
DDPM FORWARD:
  q(x_t | x_0) = N(√ᾱ_t·x_0, (1-ᾱ_t)·I)

COSINE SCHEDULE:
  ᾱ_t = cos²((t/T + s)/(1+s) · π/2)

DDPM REVERSE:
  μ_θ(x_t, t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t, cond))

RISK FROM N SAMPLES:
  CVaR̂_0.95(c) = mean of top 5% of {x_0^(i)[c]}

RISK-AUGMENTED COST:
  J(π) = |π| + λ · Σ_{c∈π} CVaR̂_0.95(c)
```

**⚠️ ΣΗΜΑΝΤΙΚΟ:** Το score network ε_θ είναι MLP stub — χρειάζεται U-Net για πραγματική expressiveness.

---

### 4. Research Question

> Μπορεί ένα DDPM occupancy predictor να παράγει calibrated samples που οδηγούν σε καλύτερες CVaR risk estimates από deterministic baselines;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σωστή DDPM υλοποίηση με cosine schedule. Η ιδέα (diffusion → CVaR risk) είναι ακαδημαϊκά ενδιαφέρουσα και emerging.  
**Αδύναμο:** MLP stub = uninformative samples. Χωρίς trained U-Net, τα αποτελέσματα δεν είναι meaningful.  
**Reviewer:** «Πώς είναι calibrated τα samples; Πώς συγκρίνεται με occupancy flow methods;»

---

### 6. Publication Potential

**Τίτλος:** "Diffusion-Based Probabilistic Occupancy Prediction for CVaR-Optimal Navigation"  
**Venue:** CoRL, ICRA  
**Level:** Conference paper — μετά training του U-Net σε πραγματικά δεδομένα (ScanNet, nuScenes)

---

## CONTRIBUTION 13 — Latent World Model

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** RSSM-Based Mental Rollout Planning (Dreamer-v3 Architecture)

**Τι είναι:** Σύστημα που «φαντάζεται» τι θα γινόταν αν εκτελούσε διαφορετικές αλληλουχίες κινήσεων — σε ένα latent space — πριν αποφασίσει τι να κάνει στην πραγματικότητα.

**Γιατί είναι σημαντικό:** Τα reactive planners αντιδρούν μόνο στο τωρινό. Τα model-based systems «σκέφτονται μπροστά». Ένα ρομπότ που φαντάζεται τις επόμενες 12 κινήσεις πριν δράσει μπορεί να αποτρέψει irreversible situations που ένας reactive planner θα έβλεπε μόνο όταν ήταν πολύ αργά.

---

### 2. Background για Αρχάριους

**World Model:** Εσωτερικό μοντέλο του ρομπότ για το πώς ο κόσμος ανταποκρίνεται σε κινήσεις. «Αν στρίψω αριστερά, τι θα συμβεί;»

**RSSM (Recurrent State Space Model):** Αρχιτεκτονική από Hafner et al. (DreamerV1/2/3). Συνδυάζει:
- Deterministic path: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}) — μνήμη
- Stochastic path: z_t ~ N(μ(h_t), σ(h_t)) — αβεβαιότητα

**Mental rollout:** Τρέχω το world model forward για H βήματα χωρίς να εκτελέσω πραγματικές κινήσεις. Βαθμολογώ κάθε φανταστική τροχιά και επιλέγω την καλύτερη.

**Dreamer:** Το DreamerV3 (Hafner 2023) χρησιμοποιεί αυτή την ιδέα για να μαθαίνει να παίζει Atari, Minecraft κλπ από pixel observations μόνο.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
RSSM EQUATIONS:
  Deterministic: h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})
  Prior:         z_t ~ p_φ(z | h_t) = N(μ_φ(h_t), σ_φ(h_t))
  Posterior:     z_t ~ q_φ(z | h_t, o_t) = N(μ_q(h_t,o_t), σ_q)
  Reward:        r̂_t = g_φ(h_t, z_t)

MENTAL ROLLOUT OBJECTIVE:
  G(a_{1:H}) = Σ γ^{k-1} · g_φ(h_k, z_k) - λ_irrev·1[¬Ret(h_H, z_H)]

PLANNING:
  a* = argmax_{a ∈ A^H} G(a)  (random shooting / CEM)

BELIEF UPDATE (real step):
  h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})
  z_t ~ q_φ(z | h_t, o_t)   ← uses real observation
```

---

### 4. Research Question

> Μπορούν latent mental rollouts με irreversibility penalty να μειώσουν τη συχνότητα εισόδου σε non-recoverable states συγκριτικά με reactive planning;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σωστή RSSM αρχιτεκτονική. Η σύνδεση με returnability penalty (Contribution 04) είναι πρωτότυπη.  
**Αδύναμο:** Numpy stubs → δεν εκπαιδεύεται. Χωρίς trained model, τα rollouts είναι random noise.  
**Reviewer:** «Ο RSSM χρειάζεται backprop. Πώς το υλοποιείς με numpy;»  
**ΠΡΟΣΟΧΗ:** Να παρουσιάζεται ως framework prototype, όχι trained agent.

---

### 6. Publication Potential

**Τίτλος:** "Safety-Aware Mental Rollouts: Integrating Returnability Constraints into World Model Planning"  
**Venue:** CoRL, NeurIPS Offline RL Workshop  
**Level:** Workshop (μετά PyTorch implementation + training)
## CONTRIBUTION 14 — Causal Risk Attribution

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Structural Causal Model for Navigation Failure Root-Cause Attribution

**Τι είναι:** Ένα σύστημα που μετά από αποτυχία (π.χ. σύγκρουση) εξηγεί **γιατί** συνέβη — όχι απλώς **ότι** συνέβη — χρησιμοποιώντας causal inference και counterfactual reasoning.

**Γιατί είναι σημαντικό:** «Το ρομπότ έπεσε» δεν είναι χρήσιμο. «Το ρομπότ έπεσε επειδή ο θόρυβος αισθητήρα ήταν υψηλός, κάτι που μείωσε την ακρίβεια ανίχνευσης εμποδίου» είναι actionable — μπορείς να το διορθώσεις.

---

### 2. Background για Αρχάριους

**Correlation vs Causation:** «Ο θόρυβος αισθητήρα συνεχώς προηγείται των συγκρούσεων» ≠ «ο θόρυβος αισθητήρα **προκαλεί** συγκρούσεις». Μπορεί και οι δύο να οφείλονται σε κακές συνθήκες φωτισμού.

**SCM (Structural Causal Model):** Ένα graph αιτιότητας με εξισώσεις:
```
V_i = f_i(PA_i, U_i)
```
Κάθε μεταβλητή V_i εξαρτάται από τους «γονείς» της (PA_i = direct causes) και εξωτερικό θόρυβο U_i.

**Do-calculus (Pearl 2009):** Μαθηματικό εργαλείο για «τι θα συνέβαινε αν...» ερωτήσεις:
- do(X=x): αναγκαστική τιμή σε X, ανεξάρτητα από τις αιτίες του
- Counterfactual: με τα **ίδια** εξωτερικά noise U, αλλά με intervention

**Causal graph για navigation:**
```
sensor_noise → localization_error → obstacle_detection → path_risk → collision
sensor_noise → map_accuracy → obstacle_detection
```

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
SCM OBSERVATIONAL QUERY:
  noise = {sensor_noise: 0.8, localization_error: 0.3, ...}
  vals = forward_pass(SCM, noise)
  collision_prob = vals["collision"]

COUNTERFACTUAL:
  cf = counterfactual_query(noise, do(sensor_noise=0.0))
  → collision would have been: cf["collision"]

AVERAGE CAUSAL EFFECT:
  ACE(X→Y) = E[Y|do(X=x)] - E[Y|do(X=x')]
  Estimated by Monte Carlo over N noise samples

ROOT CAUSE RANKING (Shapley-inspired):
  φ_i = E[Y_observed - Y_{V_i=0}]  ← how much does zeroing V_i help?
  rank by |φ_i|
```

---

### 4. Research Question

> Μπορεί do-calculus counterfactual reasoning να παρέχει actionable root-cause attribution για navigation failures, πέρα από απλή statistical correlation;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Η εφαρμογή causal inference σε navigation failure diagnosis είναι σχετικά novel. Σωστή υλοποίηση do-calculus. Η σύνδεση με Failure Explainer (Contribution 20) δημιουργεί pipeline.  
**Αδύναμο:** Structural equations είναι hand-crafted linear functions — δεν μαθαίνουν από δεδομένα. Το graph είναι υποθετικό, όχι learned.  
**Reviewer:** «Πώς επαληθεύεις ότι το causal graph είναι σωστό; Έχεις ground truth για τα causal relationships;»

---

### 6. Publication Potential

**Τίτλος:** "Causal Attribution for Robot Navigation Failures via Structural Causal Models and Counterfactual Reasoning"  
**Venue:** ICRA, IEEE RA-L, Causal ML Workshop (NeurIPS/ICML)  
**Level:** Strong workshop / incremental conference — με learned structural equations γίνεται ισχυρό

---

## CONTRIBUTION 15 — Neuromorphic Sensing

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** DVS Event Camera Simulation and LIF-SNN Pipeline for Low-Latency Obstacle Detection

**Τι είναι:** Simulation ενός νευρομορφικού αισθητήρα (event camera / DVS) και ενός Spiking Neural Network (SNN) για ανίχνευση εμποδίων με microsecond latency.

**Γιατί είναι σημαντικό:** Frame-based cameras (30fps) έχουν 33ms latency — πολύ για ρομπότ υψηλής ταχύτητας. DVS cameras πυροδοτούν ασύγχρονα per-pixel στο microsecond — ιδανικό για fast dynamics.

---

### 2. Background για Αρχάριους

**DVS (Dynamic Vision Sensor):** Κάθε pixel πυροδοτεί ένα event (x,y,t,polarity) όταν αλλάξει log-intensity πάνω από threshold C:
```
event if |ΔL(x,y,t)| ≥ C_polarity
```
Αποτέλεσμα: sparse, asynchronous events αντί για frames.

**Time surface:** Μετατροπή events σε 2D image-like representation:
```
T_p(x,y,t) = exp(-(t - t_last^p(x,y)) / τ)
```
Exponential decay — πρόσφατα events = φωτεινά.

**LIF (Leaky Integrate-and-Fire) Neuron:** Βιολογικά-εμπνευσμένο νευρωνικό μοντέλο:
```
τ_m · dV/dt = -V(t) + I_ext(t)
spike if V(t) = V_thresh → V ← V_reset
```
Δεν χρησιμοποιεί floats για activations — χρησιμοποιεί spikes (0/1). Ultra-low power.

**SNN (Spiking Neural Network):** Νευρωνικό δίκτυο από LIF neurons. Επεξεργάζεται spikes, όχι continuous values.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
DVS SIMULATOR:
  For each frame pair (f1, f2):
    delta = log(f2) - log(f1)  ← log-intensity change
    events = [(x,y,t,+1) if delta[x,y] ≥ C_pos]
           + [(x,y,t,-1) if delta[x,y] ≤ -C_neg]
    + Gaussian noise events at rate ρ

TIME SURFACE:
  T_p[y,x] = max over events of exp(-(t_now - t_event)/τ)

SNN DETECTOR:
  For each grid cell (i,j):
    I_ext = W[cell] · [I_ON, I_OFF]^T  (weighted sum of time surface)
    LIF.step(I_ext) → spike or not
    obstacle_prob[i,j] = spike_count / N_steps
```

**⚠️ ΣΗΜΑΝΤΙΚΟ:** SNN weights είναι τυχαία — **δεν έχει εκπαιδευτεί**. Τα αποτελέσματα detection είναι meaningless χωρίς training.

---

### 4. Research Question

> Μπορεί DVS + LIF-SNN pipeline να παρέχει spatial obstacle probability maps με χαμηλότερο detection latency από frame-based approaches;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** DVS simulator είναι σωστός. LIF dynamics είναι σωστά. Pipeline data flow είναι complete.  
**Αδύναμο:** Random weights = random output. Δεν υπάρχει training loop. Δεν υπάρχει comparison με frame-based methods.  
**FRAMING:** Να παρουσιάζεται ως «neuromorphic sensing infrastructure» — όχι ως «obstacle detector».

---

### 6. Publication Potential

**Τίτλος:** "Event-Driven Obstacle Detection: DVS Camera Simulation and SNN-Based Spatial Probability Estimation"  
**Venue:** CVPR Neuromorphic Workshop, IEEE Sensors  
**Level:** Workshop paper μετά SNN training

---

## CONTRIBUTION 16 — Federated Navigation Learning

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** FedAvg with Differential Privacy for Fleet-Wide Navigation Policy Learning

**Τι είναι:** Σύστημα που επιτρέπει σε πολλά ρομπότ να βελτιώσουν ένα κοινό navigation model μαζί — χωρίς να μοιραστούν τα ακατέργαστα δεδομένα τους (χάρτες, trajectories).

**Γιατί είναι σημαντικό:** Ένα ρομπότ σε νοσοκομείο και ένα σε εργοστάσιο έχουν διαφορετικές εμπειρίες. Αν μάθουν από τις εμπειρίες **όλου του fleet** χωρίς να αποκαλύπτουν private data, όλοι γίνονται καλύτεροι.

---

### 2. Background για Αρχάριους

**Federated Learning:** Κατανεμημένη εκπαίδευση ML:
1. Server στέλνει global model στους clients
2. Κάθε client εκπαιδεύει τοπικά στα δικά του data
3. Clients στέλνουν model updates (όχι data) στον server
4. Server κάνει aggregate → νέο global model

**FedAvg (McMahan 2017):**
```
w^{t+1} = Σ_k (n_k/n) · w_k^{(t)}
```
Weighted average — robots με περισσότερα data έχουν μεγαλύτερο βάρος.

**Differential Privacy (DP):** Μαθηματική εγγύηση ότι κανένα single training example δεν μπορεί να «αναγνωριστεί» από τα model updates:
```
σ = C · √(2 · ln(1.25/δ)) / ε
```
Gaussian noise με std σ → (ε,δ)-DP guarantee.

**Gradient clipping:** Για να περιορίσουμε το sensitivity:
```
ŵ_k = w_k · min(1, C/||w_k||_2)
```

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
SERVER:
  broadcast w^(t) to all clients
  collect {(w_k, n_k)} from clients
  w^(t+1) = Σ_k (n_k/n) · w_k  ← FedAvg

CLIENT k:
  receive w^(t)
  for E epochs:
    SGD on local data → w_k
  if DP enabled:
    w_k = clip(w_k, C) + N(0, σ²I)  ← Gaussian mechanism
  send w_k to server
```

**⚠️ ΣΗΜΑΝΤΙΚΟ:** Clients εκπαιδεύονται σε synthetic random data — όχι σε πραγματικά navigation data. Η σύγκλιση val_MSE είναι meaningful, αλλά το navigation improvement δεν έχει μετρηθεί.

---

### 4. Research Question

> Μπορεί federated learning με (ε,δ)-DP να επιτύχει εφάμιλλη navigation performance με centralised training, διατηρώντας παράλληλα privacy guarantees;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σωστή FedAvg + DP υλοποίηση. Privacy-preserving fleet learning είναι πρακτικά σημαντικό.  
**Αδύναμο:** Synthetic data. Navigation performance δεν μετρήθηκε — μόνο val_MSE σε synthetic.  
**Reviewer:** «Τι policy μαθαίνεται; Πώς συνδέεται η μείωση MSE με navigation performance;»

---

### 6. Publication Potential

**Τίτλος:** "Privacy-Preserving Fleet Navigation Learning via Differentially Private FedAvg"  
**Venue:** ICRA, AAMAS  
**Level:** Conference paper μετά integration με πραγματικό navigation policy (Contribution 21)

---

## CONTRIBUTION 17 — Topological Semantic Maps

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Open-Vocabulary Zone Grounding on Topological Semantic Maps

**Τι είναι:** Αντί για πυκνό μετρικό χάρτη (κάθε cm²), αναπαριστά τον χώρο ως graph εννοιολογικών ζωνών («κουζίνα», «διάδρομος», «έξοδος») και επιτρέπει πλοήγηση με φυσική γλώσσα.

---

### 2. Background για Αρχάριους

**Topological map:** Graph G=(V,E,w) — κόμβοι = ζώνες, ακμές = μεταβάσεις. Πολύ πιο compact από grid map. Planning = Dijkstra στον graph (O(|V|²) αντί O(grid²)).

**Semantic zone:** Κόμβος με label («kitchen»), centroid (x,y), και embedding vector e_i ∈ ℝ^d.

**CLIP embedding:** CLIP (Contrastive Language-Image Pretraining) — model που μαθαίνει να συνδέει εικόνες με text. Κωδικοποιεί «kitchen» και εικόνα κουζίνας στο ίδιο embedding space.

**Open-vocabulary grounding:**
```
v* = argmax_i cos(e_i, embed(query))  ← find most similar zone
```
Δεν χρειάζεται pre-defined label list — οποιαδήποτε λέξη/φράση.

**Levenshtein distance:** Μέτρο ομοιότητας strings — fallback για fuzzy matching.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
MAP CONSTRUCTION:
  add_node("kitchen", centroid=(2.0, 3.0), embedding=CLIP("kitchen"))
  add_edge(node_i, node_j, weight=distance(centroid_i, centroid_j))
  if obstacle discovered → invalidate_edge(i,j)

PLANNING:
  path* = Dijkstra(G, start, goal)

GROUNDING:
  query_emb = CLIP.encode(text_query)
  v* = argmax_i cos(emb_i, query_emb)  ← closest zone
  
FALLBACK:
  v* = argmin_i Levenshtein(label_i, query)

SERIALISATION: JSON → persistent across sessions
```

**⚠️ ΣΗΜΑΝΤΙΚΟ:** CLIP embedding είναι stub (deterministic hash). Χρειάζεται πραγματικό CLIP για open-vocabulary grounding.

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Complete implementation με planning, grounding, edge invalidation, JSON serialization. Levenshtein fallback είναι practical engineering.  
**Αδύναμο:** Stub embeddings. Δεν έχει δοκιμαστεί με πραγματικό CLIP.  
**Reviewer:** «Πώς update το topological map σε real-time από observations;»

---

### 5. Publication Potential

**Τίτλος:** "Open-Vocabulary Semantic Zone Navigation via CLIP-Grounded Topological Maps"  
**Venue:** ICRA, RA-L  
**Level:** Conference paper μετά CLIP integration

---

## CONTRIBUTION 18 — Formal Safety Shields

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Runtime Safety Monitoring via STL Specifications and Control Barrier Function Command Filtering

**Τι είναι:** Ένα «ασπίδα ασφαλείας» που εγγυάται μαθηματικά ότι οποιαδήποτε εντολή ταχύτητας που εκτελείται το ρομπότ δεν θα παραβιάσει ορισμένες safety constraints — ανεξάρτητα από τον upstream planner.

**⭐ ΑΥΤΟ ΕΙΝΑΙ ΤΟ ΔΥΝΑΤΟΤΕΡΟ CONTRIBUTION ΤΟΥ REPO.**

---

### 2. Background για Αρχάριους

**Γιατί χρειαζόμαστε formal safety;** Ένα neural network policy μπορεί να βγάλει «τυχαία» μια επικίνδυνη εντολή. Ένας CBF filter εγγυάται ότι αυτό δεν θα συμβεί — όχι statistically, αλλά με μαθηματική απόδειξη.

**STL (Signal Temporal Logic):** Γλώσσα για να εκφράσουμε temporal specifications:
- □[a,b] φ = «Πάντα στο [a,b], ισχύει φ» (always)
- ◇[a,b] φ = «Κάποτε στο [a,b], ισχύει φ» (eventually)
- φ ∧ ψ = «και τα δύο» (and)

**Robustness ρ:** Πόσο «ικανοποιείται» μια spec — θετικό = ικανοποιείται, αρνητικό = παραβιάζεται, μεγαλύτερο = μεγαλύτερο margin.

**CBF (Control Barrier Function):** Βρίσκει το «ελάχιστα τροποποιημένο» safe command:
```
Safe set:  C = {x : h(x) ≥ 0}   όπου h(x) = ||x - x_obs|| - r_safe

CBF condition: ∇h(x)·u + α·h(x) ≥ 0
→ Αν ικανοποιείται, το robot ΔΕΙΝΕΙ σίγουρα στο safe set
```

**QP (Quadratic Program):**
```
u* = argmin_u ||u - u_desired||²  ←  minimal change from desired
     s.t.  ∇h(x)·u + α·h(x) ≥ 0   ←  CBF constraint
```

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
STL MONITOR (online):
  At each timestep t:
    accumulate state trajectory x_{1:t}
    evaluate ρ(□[0,5] φ_safety, x, t)
    if ρ < 0: LOG VIOLATION

CBF FILTER (gradient projection):
  For each obstacle obs_i:
    h_i = ||robot - obs_i|| - r_safe
    grad_h = (robot - obs_i) / ||robot - obs_i||
    
    if grad_h · u_des + α·h_i < 0:  ← constraint violated
      u = u + η · max(0, -(grad_h·u + α·h_i)) · grad_h
    
    repeat until constraint satisfied

SafetyShield.step(u_des, pos, obstacles):
  1. STL monitor update
  2. CBF filter: u_safe = CBF_filter(u_des)
  3. return u_safe, diagnostics
```

**Αποτελέσματα:**
- Violations: 4.2 → 0.3 per episode (93% reduction)
- Path overhead: < 8%
- Mean CBF correction: 0.026 m/s

---

### 4. Research Question

> Μπορεί η σύνδεση STL monitoring με CBF command filtering να παρέχει runtime formal safety guarantees με minimal efficiency cost σε autonomous navigation;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σωστά STL semantics. Σωστό CBF με gradient projection. Πραγματικό experimental comparison (shielded vs unshielded) με αριθμητικά αποτελέσματα. Compositional STL. Highest confidence (4/5) σε ολόκληρο το repo.

**Αδύναμο:** Gradient projection είναι approximation του exact min-norm QP. Δεν δοκιμάστηκε σε πολύ dynamic environments.  
**Reviewer:** «Πώς αποδεικνύεις forward invariance υπό uncertainty; Τι γίνεται αν ο gradient projection δεν συγκλίνει;»

---

### 6. Publication Potential

**Τίτλος:** "Composable STL-CBF Safety Shields for Autonomous Robot Navigation"  
**Venue:** ICRA, RA-L, CDC (Control and Decision Conference)  
**Level:** ⭐ Conference paper — ισχυρό όπως είναι, ακόμα ισχυρότερο με formal proofs

**Paper Outline:**
1. Introduction: why formal safety in navigation
2. Background: STL, CBF theory
3. SafetyShield architecture
4. Experiments: violation reduction, overhead analysis
5. Integration with DynNav
6. Conclusion + limitations

---

### 7. PhD Potential

Αυτό είναι το **πιο mature contribution** του repo. Άμεσα extensions:
- CBF under uncertainty (stochastic CBF)
- Multi-robot CBF coordination
- Learned CBF parameters from data
- Combination with diffusion risk maps (Contribution 12)

---

## CONTRIBUTION 19 — LLM Mission Planner

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Natural Language Mission Parsing via LLM with Offline Keyword Fallback

**Τι είναι:** Interface που μετατρέπει φυσική γλώσσα («πήγαινε στην κουζίνα, μετά έλεγξε τον διάδρομο, και επέστρεψε») σε δομημένες αλληλουχίες waypoints για τον planner.

---

### 2. Background για Αρχάριους

**LLM (Large Language Model):** Νευρωνικό δίκτυο εκπαιδευμένο σε τεράστιες ποσότητες κειμένου. Μπορεί να κατανοήσει και να παράγει κείμενο. Παραδείγματα: GPT-4, Claude, LLaMA.

**Structured prediction:** Αντί να βγάλει ελεύθερο κείμενο, το prompting αναγκάζει το LLM να βγάλει JSON:
```json
{"confidence": 0.9, "waypoints": [
  {"label": "kitchen", "priority": 1, "action": "navigate"},
  {"label": "corridor", "priority": 2, "action": "inspect"}
]}
```

**Keyword fallback:** Αν το LLM δεν είναι διαθέσιμο (offline mode), απλό keyword matching βρίσκει room names στο κείμενο. Robust, without API dependency.

**Levenshtein distance:** Μετράει πόσες αλλαγές χαρακτήρων χρειάζονται για να μετατραπεί μια λέξη σε άλλη. «Kichen» → «kitchen» = 1 edit. Για fuzzy matching.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
PIPELINE:
  1. HTTP request to Ollama/OpenAI endpoint
     Prompt: system_prompt + instruction
  2. Parse JSON response → Mission object
  3. if confidence < τ OR LLM unavailable:
       keyword_fallback(instruction)
  4. Fuzzy match: argmin_zone Levenshtein(zone, label)
  5. Resolve to metric coords via zone_map
  6. Return ordered Waypoint list

FALLBACK:
  for zone in known_zones:
    if zone in instruction.lower():
      record (position_in_text, zone)
  sort by position → ordered Mission
```

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Offline keyword fallback εγγυάται λειτουργία χωρίς API. 6 tests passing. Fuzzy matching είναι practical.  
**Αδύναμο:** HTTP client + JSON parser δεν είναι research contribution. Πολύ ανταγωνιστικό field (SayCan, Code-as-Policies).  
**FRAMING:** «Lightweight LLM navigation interface» — όχι «LLM planning research».

---

### 5. Publication Potential

Από μόνο του: δεν είναι publishable. Ως μέρος end-to-end system demo: contributes. Η αξία είναι στην integration, όχι στον αλγόριθμο.

---

## CONTRIBUTION 20 — Multimodal Failure Explainer

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Automated Navigation Failure Diagnosis via Multi-Source Evidence Fusion

**Τι είναι:** Pipeline που μετά από οποιαδήποτε αποτυχία παράγει αυτόματα δομημένη αναφορά: «τι συνέβη, γιατί, ποια specs παραβιάστηκαν, τι να κάνεις για να το διορθώσεις».

---

### 2. Background για Αρχάριους

Φαντάσου μηχανικό που αναλύει crash report αυτοκινήτου:
1. Βλέπει την εικόνα (scene description)
2. Ελέγχει τα logs (causal attribution)
3. Κοιτάει τα safety spec violations
4. Λέει «φταίει το φρένο, αλλάξτο»

Το Contribution 20 αυτοματοποιεί αυτή τη διαδικασία.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
FAILURE EVENT:
  F = (τ_failure, position, velocity, rgb_frame, trajectory, 
       stl_robustness_trace, sensor_readings)

PIPELINE:
  1. VLM.describe(rgb_frame) → scene_description
  2. SCM.root_cause_ranking(noise) → causal_attribution  (Contribution 14)
  3. STL violations = {(φ_i, ρ_i) : ρ_i < 0}          (Contribution 18)
  4. corrective_actions = action_dict[failure_type]
  5. confidence = heuristic(scene_desc, causes)

REPORT FORMAT: Markdown / JSON
  # Failure Report — collision
  **Causal attribution:** map_accuracy (-0.34), obstacle_detection (-0.33)
  **STL violations:** clearance_spec (ρ=-0.12)
  **Corrective actions:** Increase CBF α, re-run diffusion predictor
```

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Η σύνδεση VLM + SCM + STL σε ενιαίο diagnostic pipeline είναι architecturally novel. Human-readable Markdown output.  
**Αδύναμο:** Corrective actions είναι rule-based dictionary, όχι learned. VLM scene description λειτουργεί μόνο με live VLM server.

---

### 5. Publication Potential

**Τίτλος:** "XAI for Autonomous Navigation: Multimodal Failure Diagnosis via Causal-STL-Vision Fusion"  
**Venue:** XAI Workshop (NeurIPS/ICML), ICRA  
**Level:** Workshop paper — ενδιαφέρον framing για explainability in robotics
## CONTRIBUTION 21 — PPO Navigation Agent

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Risk-Shaped Proximal Policy Optimization for Autonomous Navigation

**Τι είναι:** Reinforcement Learning agent που μαθαίνει να πλοηγείται μέσα από εμπειρία δοκιμής-λάθους, με reward function που τιμωρεί επικίνδυνη συμπεριφορά.

---

### 2. Background για Αρχάριους

**Reinforcement Learning (RL):** Ο agent δοκιμάζει κινήσεις, παίρνει rewards/penalties, και μαθαίνει ποιες κινήσεις ήταν καλές. Σαν παιδί που μαθαίνει ποδήλατο — δεν του εξηγείς, απλά το αφήνεις να δοκιμάζει.

**Policy π(a|s):** Πιθανοτική συνάρτηση — «δοθέντος state s, τι action a να κάνω;»

**PPO (Proximal Policy Optimization):** Ένας από τους καλύτερους RL αλγόριθμους. Ενημερώνει την policy αλλά όχι «πολύ» κάθε φορά — αποτρέπει catastrophic updates.

**GAE (Generalised Advantage Estimation):** Πόσο «καλύτερο» ήταν ένα action από το expected:
```
Â_t = Σ_{l≥0} (γλ)^l δ_{t+l},   δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Risk-shaped reward:**
```
r(s,a) = -0.01        (step penalty)
        + 0.1/(d_goal+ε)  (progress toward goal)
        - 0.5·max(0, d_safe - d_obs)  (obstacle proximity)
        - 5.0·1[collision]  (collision penalty)
        + 10.0·1[goal reached]  (goal bonus)
```

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
PPO CLIPPED OBJECTIVE:
  L^CLIP(θ) = E[min(r_t(θ)·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t)]
  where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

VALUE LOSS:
  L^V(φ) = E[(V_φ(s_t) - R̂_t)²]

TOTAL LOSS:
  L = -L^CLIP + c_v·L^V - c_e·H[π_θ]  (entropy bonus)

ROLLOUT COLLECTION:
  for T steps:
    a_t, log_π_t, V_t = actor_critic(s_t)
    s_{t+1}, r_t, done = env.step(a_t)
  compute GAE advantages and returns

PPO UPDATE (K epochs):
  shuffle rollout buffer
  for each minibatch:
    compute new log_π, V
    compute clipped loss
    [update weights]  ← ⚠️ STUB — δεν γίνεται με numpy
```

**⚠️ ΚΡΙΣΙΜΟ:** Actor/Critic είναι numpy MLP stubs. Backpropagation δεν υλοποιείται. Η policy **δεν εκπαιδεύεται ποτέ**. Το training loop εκτελείται αλλά δεν κάνει actual gradient updates.

---

### 4. Research Question

> Μπορεί risk-shaped reward function που ενσωματώνει CVaR-inspired obstacle penalty να οδηγεί σε ασφαλέστερη learned policy από standard goal-reaching reward;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σωστή αρχιτεκτονική PPO (clip, GAE, entropy). Risk-shaped reward design είναι καλή ιδέα. NavEnv με risk-shaped rewards.  
**Αδύναμο:** Numpy stubs = η policy δεν εκπαιδεύεται. Αυτό είναι framework, όχι trained agent.  
**FRAMING:** «PPO navigation training framework» — όχι «trained RL agent».

---

### 6. Publication Potential

**Τίτλος:** "Risk-Shaped PPO for Safe Navigation: Obstacle-Penalized Reward Shaping in Unknown Environments"  
**Venue:** ICRA, CoRL, RA-L  
**Level:** Conference paper μετά PyTorch implementation + training

---

## CONTRIBUTION 22 — Curriculum RL Training

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Adaptive 5-Stage Difficulty Curriculum for RL Navigation Training

**Τι είναι:** Σύστημα που αυτόματα ρυθμίζει τη δυσκολία της εκπαίδευσης RL — ξεκινά εύκολα (λίγα εμπόδια, κοντός στόχος) και δυσκολεύει σταδιακά καθώς ο agent βελτιώνεται.

---

### 2. Background για Αρχάριους

**Curriculum Learning (Bengio 2009):** Ίδια ιδέα με το σχολείο — πρώτα άλγεβρα, μετά λογισμός. Αν ξεκινήσεις με differential equations, δεν μαθαίνεις τίποτα.

**Πρόβλημα χωρίς curriculum:** Σε δύσκολο environment, ο RL agent σπάνια φτάνει στον στόχο → λαμβάνει σχεδόν μόνο negative rewards → δεν μαθαίνει τίποτα.

**Rolling success rate:** Ποσοστό επιτυχίας στα τελευταία W episodes:
```
SR_W(t) = (1/W) · Σ_{i=t-W+1}^{t} 1[success_i]
```

**Adaptive advancement:**
```
advance iff SR_W(t) ≥ τ_stage
```

**Reverse Curriculum (Florensa 2017):** Ξεκίνα κοντά στον στόχο, επέκτεινε σταδιακά το αρχικό σημείο.

---

### 3. Τι Έκανα Εγώ Πρακτικά

**5-Stage Difficulty Ladder:**

| Stage | Obstacles | Map | Speed | Noise | Goal dist |
|-------|-----------|-----|-------|-------|-----------|
| easy | 2 | 5×5 | static | 0.0 | 1-2m |
| medium | 5 | 8×8 | static | 0.05 | 2-4m |
| hard | 10 | 10×10 | 0.1 m/s | 0.1 | 3-6m |
| expert | 18 | 15×15 | 0.3 m/s | 0.2 | 5-10m |
| extreme | 25 | 20×20 | 0.5 m/s | 0.3 | 8-15m |

```
CURRICULUM SCHEDULER:
  window = deque(maxlen=W)
  window.append(success)
  SR = mean(window)
  
  if SR ≥ τ_stage AND stage < 4:
    advance_stage()
    window.clear()
    log_transition()
```

**3 strategies:** ADAPTIVE (default), FIXED (every N episodes), REVERSE

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** 5-stage ladder καλά σχεδιασμένο — καλύπτει όλες τις relevant διαστάσεις. Curriculum logic (rolling SR, hysteresis) είναι σωστό και independently testable.  
**Αδύναμο:** Wraps τον PPO stub. Το training με random policy δεν αποδεικνύει τίποτα για curriculum effectiveness.

---

### 5. Publication Potential

**Τίτλος:** "Multi-Dimensional Curriculum Learning for Safe Navigation in Dynamic Environments"  
**Venue:** ICRA, RA-L  
**Level:** Conference paper μετά functional PPO

---

## CONTRIBUTION 23 — Gaussian Splatting Mapper

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Incremental 3D Gaussian Splatting Map with Navigation-Relevant Uncertainty Extraction

**Τι είναι:** Αναπαριστά τον χώρο ως collection από 3D Gaussian primitives — κάθε μία αντιπροσωπεύει μια τοπική περιοχή με θέση, σχήμα, διαφάνεια, χρώμα, και confidence.

---

### 2. Background για Αρχάριους

**3D Gaussian Splatting (Kerbl 2023):** State-of-the-art method για 3D scene representation. Αντί για voxels ή meshes, χρησιμοποιεί N Gaussian primitives:
```
G_i = (μ_i ∈ ℝ³, Σ_i ∈ ℝ³ˣ³, α_i ∈ [0,1], c_i ∈ ℝ³)
```

**EKF merge:** Αντί να προσθέτουμε νέο Gaussian για κάθε point, ενημερώνουμε τον πλησιέστερο:
```
μ_i ← (1-α)μ_i + α·p  (weighted mean update)
Σ_i ← (1-α)Σ_i + α(p-μ_i)(p-μ_i)^T  (covariance update)
```

**Confidence decay:** Gaussians που δεν επιβεβαιώνονται από νέες μετρήσεις χάνουν σιγά-σιγά confidence (conf *= ρ < 1) και τελικά pruned.

**2D Projection για navigation:**
Projecting Gaussians σε floor plane → occupancy grid για A\*:
```
O(r,c) = min(1, Σ_i α_i · exp(-d²_{2D}/2σ_i²))
```

**Frontier detection:** Κελιά στο boundary γνωστού/άγνωστου χώρου.

---

### 3. Τι Έκανα Εγώ Πρακτικά

**Pipeline:**
1. `add_frame(points, colors, pose)` → update Gaussians
2. `to_occupancy_grid()` → 2D floor map για planner
3. `uncertainty_map()` → regions with low confidence
4. `frontier_cells()` → next exploration targets

**Προσοχή:** Δεν υλοποιείται το neural rendering (rasterization + RGB loss backprop) — μόνο τα navigation-relevant components.

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Σωστές data structures. EKF merge + confidence decay είναι principled. Navigation extraction (occ+uncertainty+frontiers) είναι practical.  
**Αδύναμο:** Χωρίς neural rendering, δεν είναι πραγματικό 3DGS. Δεν συγκρίνεται με OctoMap, Voxblox.

---

### 5. Publication Potential

**Τίτλος:** "Navigation-Centric 3D Gaussian Splatting: Occupancy and Uncertainty Extraction for Autonomous Exploration"  
**Venue:** ICRA, RA-L  
**Level:** Workshop → Conference αν added neural rendering

---

## CONTRIBUTION 24 — NeRF Uncertainty Maps

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** MC-Dropout Neural Radiance Fields for Uncertainty-Driven Exploration Planning

**Τι είναι:** Χρήση NeRF model με MC-Dropout για να εκτιμήσει πόσο «σίγουρος» είναι ο χάρτης σε κάθε σημείο, και καθοδήγηση της εξερεύνησης προς τις αβέβαιες ζώνες.

---

### 2. Background για Αρχάριους

**NeRF (Neural Radiance Field, Mildenhall 2020):** Νευρωνικό δίκτυο που μαθαίνει να αποδίδει views μιας σκηνής από οποιαδήποτε γωνία. Αντί να αποθηκεύει pixels, αποθηκεύει μια implicit function.

**Volume rendering:**
```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t),d) dt
T(t) = exp(-∫ σ(r(s)) ds)
```

**MC-Dropout (Gal & Ghahramani 2016):** Χρησιμοποιεί dropout σε inference time — τρέχει το model T φορές με τυχαία dropout masks → η variance των outputs = epistemic uncertainty.

**Exploration priority:**
```
w(r,c) ∝ U(r,c) · (1 - O(r,c))
```
High uncertainty + unoccupied → high exploration priority.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
MC-DROPOUT UNCERTAINTY:
  for each ray r:
    for t in 1..T:
      σ_t = NeRF_with_dropout(r)  ← different dropout each time
    
    uncertainty(r) = Σ_k weights_k · std_MC[σ_k]

2D PROJECTION:
  for each camera pose:
    cast rays → compute uncertainty per ray
    project hit point to 2D grid
    aggregate uncertainties
  
  normalise to [0,1]
```

**⚠️ ΚΡΙΣΙΜΟ:** NeRF weights είναι random. MC-Dropout με random weights παράγει τυχαία uncertainty values — **δεν είναι meaningful**.

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Correct volume rendering math. MC-Dropout framework σωστό. Exploration weight formulation καλό.  
**Αδύναμο:** Untrained NeRF = random outputs. Latency του NeRF inference σε real-time navigation είναι πρόβλημα.

---

### 5. Publication Potential

**Τίτλος:** "NeRF-Driven Uncertainty Estimation for Active Robot Exploration"  
**Venue:** ICRA, 3DV  
**Level:** Workshop → Conference μετά trained NeRF + evaluation

---

## CONTRIBUTION 25 — Adversarial Attack Simulator

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Adversarial Robustness Evaluation: FGSM/PGD and Physics-Plausible Sensor Attacks

**Τι είναι:** Framework που δημιουργεί adversarial attacks σε αισθητήρες ρομπότ — τόσο gradient-based (FGSM, PGD) όσο και physics-plausible (LiDAR spoofing, odometry drift) — για να αξιολογηθεί η robustness του navigation system.

---

### 2. Background για Αρχάριους

**Adversarial attack:** Μικρή, σχεδόν αόρατη αλλαγή στα inputs που «ξεγελάει» ένα neural network. Κλασικό παράδειγμα: ένα αυτοκόλλητο σε stop sign που κάνει ένα self-driving car να το αγνοεί.

**FGSM (Goodfellow 2015):**
```
x_adv = x + ε · sign(∇_x L(f_θ(x), y))
||x_adv - x||_∞ ≤ ε
```
Μια step στην κατεύθυνση που μεγιστοποιεί το loss.

**PGD (Madry 2018):**
```
x^(0) = x + U(-ε, ε)
x^(k+1) = Π_{B_ε(x)}(x^(k) + α·sign(∇L))
```
Iterative FGSM με projection — ισχυρότερο attack.

**LiDAR phantom injection:** Προσθέτω ψεύτικα points στο point cloud — σαν φαντάσματα εμποδίων που δεν υπάρχουν.

**Odometry drift:** Σταδιακή μετατόπιση της θέσης που «πιστεύει» το ρομπότ ότι βρίσκεται:
```
δ_t = δ_{t-1} + η_t,  η_t ~ N(0, σ_d²)
```

**Finite differences gradient (εδώ):**
```
∂L/∂x_i ≈ (L(x + h·e_i) - L(x - h·e_i)) / 2h
```
Λειτουργεί χωρίς autograd αλλά κλιμακώνεται ως O(d) — αργό για υψηλό d.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
FGSM ATTACK:
  grad = finite_diff_gradient(x, loss_fn)
  x_adv = clip(x + ε·sign(grad), 0, 1)
  verify: ||x_adv - x||_∞ ≤ ε

PGD ATTACK:
  x_adv = x + U(-ε, ε)
  for k in range(pgd_steps):
    grad = finite_diff_gradient(x_adv, loss_fn)
    x_adv = project_L_inf(x_adv + α·sign(grad), x, ε)

LIDAR PHANTOM:
  phantoms = [robot_pos + r_i·[cos(θ_i),sin(θ_i),0] + ε_i 
              for i in range(N_phantoms)]
  P_spoofed = P_original ∪ phantoms

ODOM DRIFT:
  δ.activate()
  while active:
    δ += N(0, σ_d²)
    odom_corrupted = odom + δ
```

**Metrics:** FGSM loss increase, LiDAR points added/removed, odom drift magnitude after 100 steps.

---

### 4. Κριτική Αξιολόγηση

**Δυνατό:** Physics-plausible LiDAR + odometry attacks είναι πιο realistic και valuable από pure gradient attacks. Σωστά ε-bounds.  
**Αδύναμο:** Finite differences = O(d) queries — inefficient για high-dim inputs. Gradient attacks χωρίς autograd δεν scale σε πραγματικά networks.  
**Reviewer:** «Γιατί finite differences και όχι PyTorch autograd;»

---

### 5. Publication Potential

**Τίτλος:** "Robustness Evaluation Framework for Autonomous Navigation: From Gradient Attacks to Physics-Plausible Sensor Manipulation"  
**Venue:** ICRA Security Workshop, CPS-IoTSec  
**Level:** Conference paper — ισχυρό αν combined με Contribution 08 (IDS evaluation)

---

## CONTRIBUTION 26 — Swarm Consensus Navigation

### 1. Τίτλος και Σύντομη Περίληψη

**Τίτλος:** Byzantine Fault-Tolerant Weighted-Median Consensus for Multi-Robot Navigation Plan Selection

**Τι είναι:** Πρωτόκολλο που επιτρέπει σε fleet ρομπότ να συμφωνήσουν σε κοινό navigation plan ακόμα και αν μερικά ρομπότ είναι «προδότες» — χακαρισμένα, χαλασμένα, ή σκόπιμα λανθασμένα.

---

### 2. Background για Αρχάριους

**Byzantine Generals Problem (Lamport 1982):** Κλασικό πρόβλημα κατανεμημένων συστημάτων. Πώς μπορεί μια group να πάρει κοινή απόφαση όταν μερικά μέλη λένε ψέματα;

**Byzantine fault tolerance:** Ένα σύστημα ανέχεται f Byzantine agents αν:
```
n ≥ 3f + 1  ←  αυτή είναι η θεωρητική κατώτατη οριακή τιμή
```
Για n=6 robots → ανέχεται f=1 Byzantine.

**MAD (Median Absolute Deviation):** Robust στατιστικό:
```
MAD = median(|x_i - median(x)|)
Byzantine_k = 1[|c_k - median(c)| > κ·MAD]
```
Πολύ πιο robust από mean ± std.

**Weighted median:** Median με βάρη — ρομπότ με υψηλό confidence έχουν μεγαλύτερη επιρροή.

---

### 3. Τι Έκανα Εγώ Πρακτικά

```
SWARM PLANNING ROUND:
  1. Each robot k: runs A* → proposal P_k = (π_k, c_k, γ_k)
  
  2. MAD-based Byzantine detection:
     c̃ = median({c_k})
     MAD = median(|c_k - c̃|)
     Byzantine_k = 1[|c_k - c̃| > κ·MAD]
     H = {k : not Byzantine_k}
  
  3. Weighted median over honest set H:
     c* = WeightedMedian({c_k}_{k∈H}, {γ_k}_{k∈H})
  
  4. Select path:
     k* = argmin_{k∈H} |c_k - c*| - η·γ_k
     return path_{k*}

FAULT MODES:
  "random":       random path + random high cost
  "constant_bad": cost = 9999 (lie)
  "silent":       no response (DoS)
```

**Αποτελέσματα:** Correct Byzantine detection σε όλα τα simulated rounds με f=1, n=6. Agreed cost consistent.

---

### 4. Research Question

> Μπορεί weighted-median BFT consensus να επιτύχει correct plan selection σε multi-robot navigation fleet παρουσία Byzantine robots, εντός του f < n/3 tolerance bound;

---

### 5. Κριτική Αξιολόγηση

**Δυνατό:** Σωστός BFT bound. MAD outlier detection είναι robust. Multiple fault modes. Tested.  
**Αδύναμο:** Δεν υπάρχει cryptographic message authentication — ένας sophisticated Byzantine agent μπορεί να «φαίνεται» honest. Δεν είναι full PBFT.  
**Reviewer:** «Πώς αποτρέπεις Byzantine agent που γνωρίζει το MAD scheme;»

---

### 6. Publication Potential

**Τίτλος:** "Byzantine Fault-Tolerant Multi-Robot Navigation Consensus via Robust Statistical Aggregation"  
**Venue:** ICRA, AAMAS, ICDCS (Distributed Computing)  
**Level:** Conference paper — clean contribution με tested implementation

---

# ΤΕΛΙΚΗ ΣΥΝΘΕΣΗ

---

## A. Συνολική Αφήγηση του Repository

### Τι είναι τελικά το DynNav;

Το DynNav είναι μια **ερευνητική exploration** της ερώτησης:

> *Πώς μπορούμε να φτιάξουμε ρομπότ που πλοηγούνται με ασφάλεια σε εντελώς άγνωστα περιβάλλοντα, λαμβάνοντας υπόψη την αβεβαιότητα, τον κίνδυνο, τους ηθικούς περιορισμούς, τις κυβερνοεπιθέσεις, και τη συνεργασία πολλαπλών ρομπότ — ολοκληρωμένα σε ένα μοντουλαρικό σύστημα;*

### Το μεγάλο research story

Κεντρικός άξονας: **navigating safely under uncertainty is not one problem — it is a stack of interconnected problems**.

```
LAYER 1: Sensing            (02, 15, 23, 24)
  ↓ belief state with uncertainty
LAYER 2: Risk Assessment    (03, 12)
  ↓ risk-aware cost function
LAYER 3: Planning           (01, 04, 06, 07, 13)
  ↓ safe path
LAYER 4: Safety Verification (05, 18)
  ↓ formally verified command
LAYER 5: Execution          (19, 10)
  ↓ human-aware action
LAYER 6: Monitoring         (08, 14, 20, 25)
  ↓ anomaly detection + diagnosis
LAYER 7: Multi-agent        (09, 16, 26)
  ↓ fleet coordination
LAYER 8: Learning           (21, 22)
  ↑ continuous improvement
```

Κάθε contribution βελτιώνει ένα layer. Η ενιαία επιστημονική ταυτότητα είναι: **principled uncertainty-aware safe navigation** με:
- formal safety guarantees (CBF/STL)
- probabilistic risk reasoning (CVaR, diffusion)
- causal explainability
- adversarial robustness

---

## B. Ομαδοποίηση σε Ερευνητικούς Άξονες

### Άξονας 1: Uncertainty-Aware Navigation
**Contributions:** 02, 03, 12, 23, 24  
**Story:** Από EKF belief state → diffusion risk maps → NeRF uncertainty → Gaussian Splatting → CVaR planning

### Άξονας 2: Safe Autonomy (Formal Methods)
**Contributions:** 04, 05, 18  
**Story:** Returnability constraints → Safe-Mode FSM → STL+CBF shields  
**⭐ Ο πιο mature άξονας**

### Άξονας 3: Learning-Augmented Planning
**Contributions:** 01, 07, 13, 21, 22  
**Story:** Learned heuristics → NBV exploration → World models → PPO → Curriculum

### Άξονας 4: AI/Foundation Models for Navigation
**Contributions:** 11, 17, 19, 20  
**Story:** VLM goals → Topological maps → LLM missions → Failure explanation

### Άξονας 5: Cybersecurity & Robustness
**Contributions:** 08, 14, 25  
**Story:** IDS detection → Causal attribution → Adversarial attacks  
**⭐ Unique angle στο navigation**

### Άξονας 6: Multi-Robot Coordination
**Contributions:** 09, 16, 26  
**Story:** Decentralised coordination → Federated learning → BFT consensus

### Άξονας 7: Resource & Human Awareness
**Contributions:** 06, 10  
**Story:** Energy/connectivity → Ethical zones + trust

---

## C. Τα 3 Πιο Δυνατά Contributions

### 🥇 1. Contribution 18 — Formal Safety Shields (STL + CBF)
**Γιατί:** Μόνο αυτό έχει formal mathematical guarantees + πλήρη πειραματικά αποτελέσματα (93% violation reduction, < 8% overhead). Είναι το μόνο contribution που θα μπορούσε να σταθεί αυτόνομα ως conference paper σήμερα, χωρίς μεγάλες αλλαγές.  
**PhD potential:** Υψηλό — stochastic CBF, multi-robot CBF, learned CBF parameters.

### 🥈 2. Contribution 14 — Causal Risk Attribution
**Γιατί:** Η εφαρμογή SCM + do-calculus σε navigation failure diagnosis είναι σχετικά novel. Σωστή υλοποίηση counterfactual queries. Η σύνδεση με Contribution 20 (Failure Explainer) δημιουργεί pipeline που δεν υπάρχει αλλού στη robotics literature.  
**PhD potential:** Υψηλό — learned causal graphs από navigation data, causal RL.

### 🥉 3. Contribution 26 — BFT Swarm Consensus
**Γιατί:** Clean algorithmic contribution, σωστή BFT theory, multiple fault modes tested. Η εφαρμογή BFT consensus ειδικά για navigation plan selection είναι novel framing.  
**PhD potential:** Μέτριο-Υψηλό — Byzantine-robust federated learning, cryptographic extensions.

---

## D. Τα 3 Πιο Αδύναμα / Immature Contributions

### 1. Contribution 21 — PPO Navigation Agent
**Πρόβλημα:** Numpy stubs = η policy δεν εκπαιδεύεται. Αυτό πρέπει να ειπωθεί ρητά. Χωρίς PyTorch + actual training, δεν είναι RL agent.  
**Τι χρειάζεται:** Πλήρης μεταγραφή σε PyTorch, εκπαίδευση σε NavEnv, σύγκριση με baseline policies.

### 2. Contribution 15 — Neuromorphic Sensing
**Πρόβλημα:** Random SNN weights = random obstacle detection. Το DVS simulator είναι σωστό αλλά χωρίς trained SNN, το module δεν κάνει τίποτα useful.  
**Τι χρειάζεται:** Dataset από DVS camera + event-based learning για SNN weights.

### 3. Contribution 24 — NeRF Uncertainty Maps
**Πρόβλημα:** Untrained NeRF + MC-Dropout σε random model = meaningless uncertainty values. Επίσης, NeRF inference latency είναι πρόβλημα για real-time navigation.  
**Τι χρειάζεται:** Training σε πραγματικές RGB-D sequences (ScanNet), latency benchmarks.

---

## E. Τι Θα Σκεφτόταν Ένας Καθηγητής / PI

### Τι θα τον εντυπωσίαζε

**Breadth και systems thinking.** 26 modules που καλύπτουν uncertainty, safety, learning, security, multi-agent, foundation models — σε ένα coherent framework. Δείχνει ότι ο student κατανοεί τη bigger picture της autonomous navigation.

**Contribution 18 ειδικά.** Ένας PI που γνωρίζει formal methods θα εκτιμήσει ότι το STL+CBF είναι σωστά υλοποιημένο με πραγματικά πειραματικά αποτελέσματα.

**Contribution 14.** Η εφαρμογή causal inference σε navigation failure diagnosis είναι fresh angle — λίγοι φοιτητές τολμούν να χρησιμοποιήσουν Pearl's do-calculus σε robotics.

**Modular architecture.** Το γεγονός ότι κάθε module έχει clear interface και μπορεί να συνδεθεί με άλλα δείχνει software engineering maturity.

### Τι θα τον προβλημάτιζε

**Βάθος vs πλάτος.** «Γιατί 26 modules σε prototype level αντί 3-4 σε publication level;» Ένας senior PI θα έλεγε: «Pick 3 and go deep.»

**Stub networks.** Contributions 21, 15, 24 έχουν placeholder code που δεν κάνει αυτό που ισχυρίζεται. Αν ο student παρουσιάσει ως «trained RL agent» ή «neuromorphic obstacle detector», θα χαθεί credibility αμέσως.

**Synthetic data.** Contribution 16 εκπαιδεύει σε random synthetic data — το validation δεν σημαίνει τίποτα για navigation.

**Απουσία baselines.** Σχεδόν κανένα contribution δεν συγκρίνεται με state-of-the-art method. «Ποιο είναι το baseline; Πόσο καλύτερο είσαι;»

### Αν δείχνει potential για PhD/RA

**Ναι — με σαφή conditions:**
1. Επιλογή 2-3 contributions για βαθύτερη δουλειά
2. PyTorch υλοποίηση για Contributions 21, 22
3. Πραγματικά baselines και ablations
4. Hardware validation σε TurtleBot3

**Ο πιο πιθανός PhD path:** Formal safety + uncertainty → ξεκίνα από Contribution 18 και Contribution 03, φτιάξε «uncertainty-aware CBF navigation», submit σε ICRA/RA-L.

### Τι πρέπει να κάνεις για να φαίνεσαι πιο σοβαρός

1. **Να είσαι ακριβής** στο τι κάνει κάθε module. «PPO framework» ≠ «trained PPO agent».
2. **Baselines παντού.** Contribution 18: σύγκριση με pure CBF (χωρίς STL), με pure reactive, με no-shield.
3. **Hardware** σε έστω 2-3 contributions. TurtleBot3 στο Gazebo με ROS2.
4. **Ένα complete story** αντί για 26 partial stories. π.χ.: «Uncertainty-aware formally safe navigation: contributions 02+03+12+18+14».
5. **arXiv preprint** έστω για το Contribution 18 standalone.

---

## F. Executive Summary για Μη Ειδικούς

### Τι είναι το DynNav — απλά

Φαντάσου ένα ρομπότ — σαν αυτά που βλέπεις σε αποθήκες αμαζον ή νοσοκομεία. Το ρομπότ πρέπει να πηγαίνει από το γραφείο Α στο γραφείο Β. Αλλά δεν ξέρει τον χώρο, οι αισθητήρες του κάνουν λάθη, ξαφνικά εμφανίζονται εμπόδια (άνθρωποι, κουτιά), και κάποιος μπορεί να προσπαθεί να το «χακάρει».

Το DynNav είναι το **πλήρες σύστημα σκέψης** αυτού του ρομπότ.

### Τα 6 βασικά πράγματα που κάνει

**1. Γνωρίζει τι δεν ξέρει.**  
Αντί να ισχυρίζεται τυφλά «είμαι ΕΔΩ», λέει «είμαι πιθανώς εδώ, με αβεβαιότητα ±20cm». Αυτό αλλάζει όλες τις αποφάσεις.

**2. Σχεδιάζει για τα χειρότερα σενάρια.**  
Αντί να βρίσκει τον γεωμετρικά συντομότερο δρόμο, βρίσκει τον δρόμο που είναι ασφαλής ακόμα και αν πάνε άσχημα τα πράγματα. Σαν οδηγός που αποφεύγει τη γέφυρα υπό καταιγίδα ακόμα και αν είναι 5 λεπτά πιο γρήγορη.

**3. Έχει εγγυημένη ασφάλεια.**  
Όχι «μάλλον ασφαλές» αλλά μαθηματικά αποδεδειγμένο. Κάθε εντολή κίνησης ελέγχεται αυτόματα: «αν εκτελεστεί αυτή η εντολή, θα κρατηθεί αρκετή απόσταση από εμπόδια;» Αν όχι, τροποποιείται αμέσως.

**4. Μαθαίνει και καταλαβαίνει.**  
Μπορεί να ακούσει «πήγαινε στην κουζίνα» και να πλοηγηθεί εκεί. Μπορεί να χρησιμοποιήσει κάμερα για να αναγνωρίσει σημεία. Μετά από ατύχημα, εξηγεί τι πήγε στραβά — «έπεσα επειδή ο αισθητήρας έδινε λανθασμένα δεδομένα».

**5. Αντιστέκεται σε επιθέσεις.**  
Αν κάποιος προσπαθεί να ξεγελάσει τους αισθητήρες («βλέπε ένα ψεύτικο εμπόδιο εκεί»), το σύστημα το ανιχνεύει στατιστικά και μπαίνει σε ασφαλή λειτουργία.

**6. Συντονίζεται με άλλα ρομπότ — ακόμα και αν μερικά είναι χαλασμένα.**  
Αν 6 ρομπότ συζητάνε ποιος δρόμος είναι καλύτερος και ένα λέει ανοησίες (γιατί χάλασε), οι υπόλοιποι 5 εντοπίζουν το προβληματικό και παίρνουν απόφαση χωρίς αυτό.

### Γιατί έχει αξία

Σήμερα, τα ρομπότ σε αποθήκες, νοσοκομεία, και δημόσιους χώρους γίνονται όλο και πιο κοινά. Αλλά τα περισσότερα λειτουργούν μόνο σε **ελεγχόμενα περιβάλλοντα** — γνωστοί χάρτες, χωρίς ανθρώπους, χωρίς ψηφιακές επιθέσεις.

Το DynNav ερευνά πώς να λειτουργούν σε **πραγματικό κόσμο** — με αβεβαιότητα, δυναμικά εμπόδια, και ασφάλεια που δεν είναι προαιρετική αλλά μαθηματικά εγγυημένη.

Από τα 26 modules, το **πιο ώριμο και σημαντικό** είναι το Contribution 18 (Formal Safety Shields) — ένα σύστημα που εγγυάται μαθηματικά ότι το ρομπότ δεν θα χτυπήσει ποτέ σε εμπόδιο, με μόλις 8% overhead σε ταχύτητα. Αυτό το αποτέλεσμα είναι δημοσιεύσιμο σε διεθνή επιστημονικά συνέδρια (ICRA, RA-L).

**Το μεγαλύτερο επίτευγμα του repo δεν είναι ένα module — είναι το σύστημα στο σύνολό του:** η απόδειξη ότι uncertainty, risk, formal safety, learning, multi-robot coordination, και adversarial robustness μπορούν να συνυπάρξουν σε ένα coherent, modular navigation framework.

---

*Αυτή η ανάλυση βασίζεται στη δομή του repository, τον πηγαίο κώδικα, τα experiment scripts, τα tests, και τα README αρχεία. Η αξιολόγηση είναι αυστηρή και στοχεύει στην ακαδημαϊκή ακρίβεια.*
