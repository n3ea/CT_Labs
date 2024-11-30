import numpy as np
import math
from itertools import combinations, product

# Генерация всех бинарных векторов длины `cols`
def generate_binary_vectors(cols):
    return list(product([0, 1], repeat=cols))

# Вычисление значения функции f для заданных индексов вектора
def compute_f_value(vector, indices):
    return np.prod([(vector[idx] + 1) % 2 for idx in indices])

# Формирование вектора для заданных индексов и количества столбцов
def create_f_vector(indices, num_cols):
    if not indices:  # Если индексы пусты, возвращаем единичный вектор
        return np.ones(2 ** num_cols, dtype=int)
    return [compute_f_value(binary_vector, indices) for binary_vector in generate_binary_vectors(num_cols)]

# Генерация всех подмножеств для комбинаций размера от 0 до r из m столбцов
def generate_combinations_up_to_r(m, r):
    return [subset for subset_size in range(r + 1) for subset in combinations(range(m), subset_size)]

# Вычисление общего размера для матрицы Рида-Маллера
def calculate_rm_matrix_size(r, m):
    return sum(math.comb(m, i) for i in range(r + 1))

# Построение матрицы Рида-Маллера для заданных r и m
def build_rm_matrix(r, m):
    size = calculate_rm_matrix_size(r, m)
    matrix = np.zeros((size, 2 ** m), dtype=int)
    for row, subset in enumerate(generate_combinations_up_to_r(m, r)):
        matrix[row] = create_f_vector(subset, m)
    return matrix

# Сортировка комбинаций для декодирования по длине индексов
def sort_combinations_for_decoding(m, r):
    index_combinations = list(combinations(range(m), r))
    index_combinations.sort(key=len)
    return np.array(index_combinations, dtype=int)

# Создание вектора H для заданных индексов
def generate_vector_H(indices, m):
    return [binary_vector for binary_vector in generate_binary_vectors(m) if compute_f_value(binary_vector, indices) == 1]

# Получение дополнения к набору индексов
def get_complementary_indices(indices, m):
    return [i for i in range(m) if i not in indices]

# Вычисление функции f с учетом вектора ошибок
def compute_f_with_error(binary_vector, indices, t):
    return np.prod([(binary_vector[j] + t[j] + 1) % 2 for j in indices])

# Создание вектора с учетом вектора ошибок
def create_f_vector_with_error(indices, m, t):
    if not indices:
        return np.ones(2 ** m, dtype=int)
    return [compute_f_with_error(binary_vector, indices, t) for binary_vector in generate_binary_vectors(m)]

# Основной алгоритм мажоритарного декодирования
def majoritarian_decoding(received_word, r, m, matrix_size):
    word = received_word.copy()
    decoded_vector = np.zeros(matrix_size, dtype=int)
    max_weight = 2 ** (m - r - 1) - 1
    index = 0

    for i in range(r, -1, -1):
        for indices in sort_combinations_for_decoding(m, i):
            max_count = 2 ** (m - i - 1)
            zero_count, one_count = 0, 0
            complement = get_complementary_indices(indices, m)

            for t in generate_vector_H(indices, m):
                V = create_f_vector_with_error(complement, m, t)
                c = np.dot(word, V) % 2
                zero_count += (c == 0)
                one_count += (c == 1)

            if zero_count > max_weight and one_count > max_weight:
                return None  # Слишком большая неопределенность

            if zero_count > max_count:
                decoded_vector[index] = 0
            elif one_count > max_count:
                decoded_vector[index] = 1
                word = (word + create_f_vector(indices, m)) % 2
            index += 1

    return decoded_vector

# Генерация слова с ошибками для теста
def generate_word_with_errors(G, error_count):
    u = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    print("Исходное сообщение:", u)
    encoded_word = np.dot(u, G) % 2
    error_positions = np.random.choice(len(encoded_word), size=error_count, replace=False)
    encoded_word[error_positions] = (encoded_word[error_positions] + 1) % 2
    return encoded_word

# Эксперимент с одним ошибочным битом
def run_single_error_test(G):
    error_word = generate_word_with_errors(G, 1)
    print("Слово с одной ошибкой:", error_word)
    decoded_word = majoritarian_decoding(error_word, 2, 4, len(G))
    if decoded_word is None:
        print("\nНеобходима повторная отправка сообщения")
    else:
        print("Исправленное слово:", decoded_word)
        result = np.dot(decoded_word, G) % 2
        print("Результат умножения исправленного слова на матрицу G:", result)

# Эксперимент с двумя ошибочными битами
def run_double_error_test(G):
    error_word = generate_word_with_errors(G, 2)
    print("Слово с двумя ошибками:", error_word)
    decoded_word = majoritarian_decoding(error_word, 2, 4, len(G))
    if decoded_word is None:
        print("\nНеобходима повторная отправка сообщения")
    else:
        print("Исправленное слово:", decoded_word)
        result = np.dot(decoded_word, G) % 2
        print("Результат умножения исправленного слова на матрицу G:", result)

# Основная программа
if __name__ == "__main__":
    # Создание матрицы Рида-Маллера для кода (r=2, m=4)
    G_matrix = build_rm_matrix(2, 4)
    print("Порождающая матрица Рида-Маллера G:\n", G_matrix)
    run_single_error_test(G_matrix)
    run_double_error_test(G_matrix)
