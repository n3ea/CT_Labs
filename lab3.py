import numpy as np

r = 3
n = 2**r - 1
k = n - r

# Создание матрицы P с первой строкой из единиц
X = np.zeros((k, r), dtype=int)

for i in range(r):
    for j in range(k):
        if (j >> i) & 1:
            X[j, i] = 1
X[0, :] = 1
# Создание единичной матрицы I_k
I_k = np.eye(k, dtype=int)

# Порождающая матрица G
G = np.hstack((I_k, X))

print('G = \n', G)
H = np.vstack((X, np.eye(n-k)))
print('H = \n',H)

# Создаем таблицу синдромов
syndrome_map = {}

for i in range(H.shape[0]):
    error = np.zeros(H.shape[0])
    error[i] = 1
    
    syndrome = np.dot(error, H) % 2
    
    syndrome_map[tuple(syndrome)] = tuple(error)

for syndrome, error in syndrome_map.items():
    print(f"Синдром: {syndrome} Ошибка: {error}")

word = np.zeros(G.shape[0])
word[0] = 1
# word = np.array([1, 0, 0, 1])

code_word = np.dot(word, G) % 2

error_index = 0
code_word_with_error = code_word.copy()
code_word_with_error[error_index] = (code_word_with_error[error_index] + 1) % 2

# Вычисляем синдром
syndrome = np.dot(code_word_with_error, H) % 2
print(syndrome)
corrected_code_word = None
if tuple(syndrome) in syndrome_map:
    corrected_code_word = code_word_with_error.copy()
    for i, bit in enumerate(syndrome_map[tuple(syndrome)]):
        corrected_code_word[i] = (corrected_code_word[i] + bit) % 2
else:
    print("Ошибка не может быть исправлена.")

# Убеждаемся в правильности полученного слова
if np.array_equal(code_word, corrected_code_word):
    print("Ошибку успешно исправлено.")
else:
    print("Ошибка не была исправлена.")

print("Исходное кодовое слово:", code_word)
print("Кодовое слово с ошибкой:", code_word_with_error)
print("Исправленное кодовое слово:", corrected_code_word)

# Проверка для двухкратных ошибок
code_word = np.dot(word, G) % 2

# Вносим двухкратную ошибку
error_index_1 = 1
error_index_2 = 2
code_word_with_error = code_word.copy()
code_word_with_error[error_index_1] = (code_word_with_error[error_index_1] + 1) % 2
code_word_with_error[error_index_2] = (code_word_with_error[error_index_2] + 1) % 2

syndrome = np.dot(code_word_with_error, H) % 2
print(syndrome)
# Исправляем ошибку, используя таблицу синдромов
corrected_code_word = None
if tuple(syndrome) in syndrome_map:
    corrected_code_word = code_word_with_error.copy()
    for i, bit in enumerate(syndrome_map[tuple(syndrome)]):
        corrected_code_word[i] = (corrected_code_word[i] + bit) % 2
else:
    print("2. Ошибка не может быть исправлена.")

# Убеждаемся в правильности полученного слова
if np.array_equal(code_word, corrected_code_word):
    print("2. Ошибку успешно исправлено.")
else:
    print("2. Ошибка не была исправлена.")

print("2. Исходное кодовое слово:", code_word)
print("2. Кодовое слово с ошибкой:", code_word_with_error)
print("2. Исправленное кодовое слово:", corrected_code_word)

# Проверка для трёхкратных ошибок
code_word = np.dot(word, G) % 2

error_index_1 = 0
error_index_2 = 1
error_index_3 = 2
code_word_with_error = code_word.copy()
code_word_with_error[error_index_1] = (code_word_with_error[error_index_1] + 1) % 2
code_word_with_error[error_index_2] = (code_word_with_error[error_index_2] + 1) % 2
code_word_with_error[error_index_3] = (code_word_with_error[error_index_2] + 1) % 2

# Вычисляем синдром
syndrome = np.dot(code_word_with_error, H) % 2
print(syndrome)
# Исправляем ошибку, используя таблицу синдромов
corrected_code_word = None
if tuple(syndrome) in syndrome_map:
    corrected_code_word = code_word_with_error.copy()
    for i, bit in enumerate(syndrome_map[tuple(syndrome)]):
        corrected_code_word[i] = (corrected_code_word[i] + bit) % 2
else:
    print("3. Ошибка не может быть исправлена.")

# Убеждаемся в правильности полученного слова
if np.array_equal(code_word, corrected_code_word):
    print("3. Ошибку успешно исправлено.")
else:
    print("3. Ошибка не была исправлена.")

print("3. Исходное кодовое слово:", code_word)
print("3. Кодовое слово с ошибкой:", code_word_with_error)
print("3. Исправленное кодовое слово:", corrected_code_word)

# 3.3 Написать функцию формирования порождающей и проверочной
# матриц расширенного кода Хэмминга (𝟐𝒓,𝟐𝒓 −𝒓−𝟏,𝟑)  на основе
# параметра 𝒓, а также таблицы синдромов для всех однократных
# ошибок
zeros_row = np.zeros((1, H.shape[1]))

H_zeros = np.vstack((H, zeros_row))

ones_column = np.ones((H_zeros.shape[0], 1))

H_Star = np.hstack((H_zeros, ones_column))
print('H* = \n',H_Star)

sums = []

for row in G:
    row_sum = sum(row)
    sums.append(row_sum)

print("sum = ", sums)

b = [1 if num % 2 != 0 else 0 for num in sums]

G_Star = np.column_stack((G, b))

print("G* = \n", G_Star)

# Таблица синдромов для расширенных матриц Хэмминга
extended_syndrome_map = {}

for i in range(H_Star.shape[0]):
    error = np.zeros(H_Star.shape[0])
    error[i] = 1
    
    syndrome = np.dot(error, H_Star) % 2
    
    extended_syndrome_map[tuple(syndrome)] = tuple(error)

for syndrome, error in extended_syndrome_map.items():
    print(f"Синдром: {syndrome} Ошибка: {error}")

# Провести исследование расширенного кода Хэмминга для одно-,
# двух-, трёх- и четырёхкратных ошибок для 𝒓=𝟐,𝟑,𝟒.
word = np.zeros(G_Star.shape[0])
word[0] = 1
# Формируем кодовое слово длины n
code_word = np.dot(word, G_Star) % 2

# Вносим однократную ошибку
error_index = 0
code_word_with_error = code_word.copy()
code_word_with_error[error_index] = (code_word_with_error[error_index] + 1) % 2

# Вычисляем синдром
syndrome = np.dot(code_word_with_error, H_Star) % 2
print(syndrome)
# Исправляем ошибку, используя таблицу синдромов
corrected_code_word = None
if tuple(syndrome) in extended_syndrome_map:
    corrected_code_word = code_word_with_error.copy()
    for i, bit in enumerate(extended_syndrome_map[tuple(syndrome)]):
        corrected_code_word[i] = (corrected_code_word[i] + bit) % 2
else:
    print("Ошибка не может быть исправлена.")

# Убеждаемся в правильности полученного слова
if np.array_equal(code_word, corrected_code_word):
    print("Ошибку успешно исправлено.")
else:
    print("Ошибка не была исправлена.")

print("Исходное кодовое слово:", code_word)
print("Кодовое слово с ошибкой:", code_word_with_error)
print("Исправленное кодовое слово:", corrected_code_word)

# Проверка для двухкратных ошибок
code_word = np.dot(word, G_Star) % 2

# Вносим двухкратную ошибку
error_index_1 = 0
error_index_2 = 1
code_word_with_error = code_word.copy()
code_word_with_error[error_index_1] = (code_word_with_error[error_index_1] + 1) % 2
code_word_with_error[error_index_2] = (code_word_with_error[error_index_2] + 1) % 2

syndrome = np.dot(code_word_with_error, H_Star) % 2
print(syndrome)
# Исправляем ошибку, используя таблицу синдромов
corrected_code_word = None
if tuple(syndrome) in extended_syndrome_map:
    corrected_code_word = code_word_with_error.copy()
    for i, bit in enumerate(extended_syndrome_map[tuple(syndrome)]):
        corrected_code_word[i] = (corrected_code_word[i] + bit) % 2
else:
    print("2. Ошибка не может быть исправлена.")

# Убеждаемся в правильности полученного слова
if np.array_equal(code_word, corrected_code_word):
    print("2. Ошибку успешно исправлено.")
else:
    print("2. Ошибка не была исправлена.")

print("2. Исходное кодовое слово:", code_word)
print("2. Кодовое слово с ошибкой:", code_word_with_error)
print("2. Исправленное кодовое слово:", corrected_code_word)

# Проверка для трёхкратных ошибок
code_word = np.dot(word, G_Star) % 2

# Вносим двухкратную ошибку
error_index_1 = 0
error_index_2 = 1
error_index_3 = 2
code_word_with_error = code_word.copy()
code_word_with_error[error_index_1] = (code_word_with_error[error_index_1] + 1) % 2
code_word_with_error[error_index_2] = (code_word_with_error[error_index_2] + 1) % 2
code_word_with_error[error_index_3] = (code_word_with_error[error_index_2] + 1) % 2

# Вычисляем синдром
syndrome = np.dot(code_word_with_error, H_Star) % 2
print(syndrome)
corrected_code_word = None
if tuple(syndrome) in extended_syndrome_map:
    corrected_code_word = code_word_with_error.copy()
    for i, bit in enumerate(extended_syndrome_map[tuple(syndrome)]):
        corrected_code_word[i] = (corrected_code_word[i] + bit) % 2
else:
    print("3. Ошибка не может быть исправлена.")

# Убеждаемся в правильности полученного слова
if np.array_equal(code_word, corrected_code_word):
    print("3. Ошибку успешно исправлено.")
else:
    print("3. Ошибка не была исправлена.")

print("3. Исходное кодовое слово:", code_word)
print("3. Кодовое слово с ошибкой:", code_word_with_error)
print("3. Исправленное кодовое слово:", corrected_code_word)
