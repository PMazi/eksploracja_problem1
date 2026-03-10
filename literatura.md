# System rekomendacyjny – notatki (SVD, SVD++, ALS)

## Źródła

* [https://medium.com/@mohamedelamartyelamarty/svd-in-recommender-systems-concepts-math-and-code-fbb9e4685b9f](https://medium.com/@mohamedelamartyelamarty/svd-in-recommender-systems-concepts-math-and-code-fbb9e4685b9f)
* [https://ujangriswanto08.medium.com/a-beginners-guide-to-using-svd-for-building-recommender-systems-878de4b66992](https://ujangriswanto08.medium.com/a-beginners-guide-to-using-svd-for-building-recommender-systems-878de4b66992)
* [https://github.com/gbolmier/funk-svd](https://github.com/gbolmier/funk-svd)
* [https://ace.ewapub.com/article/view/10786.pdf](https://ace.ewapub.com/article/view/10786.pdf)
* [https://ijournals.in/wp-content/uploads/2025/06/1.IJSHRE-130507-Dheeraj.pdf](https://ijournals.in/wp-content/uploads/2025/06/1.IJSHRE-130507-Dheeraj.pdf)

---

# 1. Problem rekomendacji

W systemach rekomendacyjnych mamy zwykle **dużą macierz ocen**:

```
        Film1 Film2 Film3 Film4 ...
User1     5     ?     3
User2     ?     4     ?
User3     2     ?     5
```

* wiersze → użytkownicy
* kolumny → filmy
* wartości → oceny

Problem:
**większość macierzy jest pusta** (użytkownik nie ogląda wszystkiego).

To zjawisko nazywa się **sparsity (rzadkość danych)**.

Przykład z datasetu MovieLens:

| Dataset | Sparsity   |
| ------- | ---------- |
| 100K    | 93.69%     |
| 1M      | 95.75%     |
| 10M     | ~98.7%     |
| 20M     | **99.47%** |

Czyli np. dla MovieLens 20M **ponad 99% macierzy to brak danych**.

---

# 2. Matrix Factorization (faktoryzacja macierzy)

Aby rozwiązać problem brakujących ocen stosuje się **matrix factorization**.

Idea:

Zamiast trzymać ogromną macierz użytkownicy × filmy, rozkładamy ją na **dwie mniejsze macierze cech ukrytych (latent factors)**.

```
R ≈ P × Q
```

gdzie:

* **R** – macierz ocen
* **P** – cechy użytkowników
* **Q** – cechy filmów

---

# 3. Cechy ukryte (latent factors)

Cechy ukryte to **automatycznie odkryte preferencje**.

Przykład interpretacji:

### Macierz użytkowników (P)

| user | akcja | romans | horror |
| ---- | ----- | ------ | ------ |
| u1   | 0.9   | 0.1    | 0.3    |
| u2   | 0.2   | 0.8    | 0.1    |

czyli np.

* użytkownik 1 bardzo lubi **akcję**
* użytkownik 2 bardzo lubi **romanse**

---

### Macierz filmów (Q)

| film | akcja | romans | horror |
| ---- | ----- | ------ | ------ |
| f1   | 0.8   | 0.1    | 0.2    |
| f2   | 0.1   | 0.9    | 0.1    |

czyli np.

* film 1 → dużo akcji
* film 2 → dużo romansu

---

# 4. Klasyczne SVD (Funk SVD)

Predykcja oceny jest liczona jako:

```
r̂_ui = μ + b_u + b_i + q_i · p_u
```

gdzie:

* **μ** – średnia globalna ocen
* **b_u** – bias użytkownika
* **b_i** – bias filmu
* **p_u** – wektor cech użytkownika
* **q_i** – wektor cech filmu
* **q_i · p_u** – iloczyn skalarny (dopasowanie)

---

## Bias – co to jest?

Bias oznacza **odchylenie od średniej**.

Przykłady:

* użytkownik daje zwykle wysokie oceny → dodatni bias
* film jest ogólnie słabo oceniany → ujemny bias

To pomaga modelowi być bardziej realistycznym.

---

# 5. SVD++

SVD++ to ulepszona wersja klasycznego SVD.

Najważniejsza różnica:

**uwzględnia implicit feedback (dane niejawne).**

---

## Implicit feedback

To informacje typu:

* użytkownik obejrzał film
* użytkownik kliknął film
* użytkownik ocenił film

Nawet jeśli ocena była **1 gwiazdką**, to nadal jest informacja:

> użytkownik był na tyle zainteresowany, żeby obejrzeć film.

---

## Wzór SVD++

Predykcja:

[
\hat r_{ui} = \mu + b_u + b_i + q_i^T
\left(p_u + |N(u)|^{-0.5} \sum_{j \in N(u)} y_j \right)
]

gdzie:

* **N(u)** – zbiór filmów ocenionych przez użytkownika
* **y_j** – wektor cech dla implicit feedback

Interpretacja:

model bierze pod uwagę:

1. **preferencje użytkownika**
2. **cechy filmu**
3. **filmy, które użytkownik już oglądał**

Dzięki temu rekomendacje są **dokładniejsze**.

---

# 6. ALS (Alternating Least Squares)

ALS to inny popularny algorytm faktoryzacji macierzy.

Idea:

zamiast trenować wszystko naraz, algorytm **naprzemiennie optymalizuje**:

1. macierz użytkowników
2. macierz filmów

schemat:

```
1. losujemy macierz filmów
2. liczymy macierz użytkowników
3. aktualizujemy macierz filmów
4. powtarzamy
```

czyli:

```
fix movies → update users
fix users → update movies
```

Dlatego nazywa się **Alternating Least Squares**.

---

## Zalety ALS

* dobrze działa na **bardzo dużych datasetach**
* łatwo się **równolegli (Spark)**

