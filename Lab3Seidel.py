import customtkinter as ctk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from fontTools.varLib.models import nonNone
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

iteration_count = 0
final_accuracy = 0
final_residual = 0

def true_solution(x, y):
    return x**2 + y**3 + 1


def f(xi, yj):
    return 2+6*yj


def mu1(yj, a):
    return true_solution(a, yj)


def mu2(yj, b):
    return true_solution(b, yj)


def mu3(xi, c):
    return true_solution(xi, c)


def mu4(xi, d):
    return true_solution(xi, d)


def buildVector(n, m, a, b, c, d):
    vec = np.zeros((n - 1) * (m - 1))

    h = (b - a) / n
    k = (d - c) / m

    for j in range(1, m):
        for i in range(1, n):
            numberEq = (j - 1) * (n - 1) + i - 1
            vec[numberEq] -= f(i * h, j * k)

            if i == 1:
                vec[numberEq] -= mu1(j * k, a) / h ** 2

            if i == n - 1:
                vec[numberEq] -= mu2(j * k, b) / h ** 2

            if j == 1:
                vec[numberEq] -= mu3(i * h, c) / k ** 2

            if j == m - 1:
                vec[numberEq] -= mu4(i * h, d) / k ** 2

    return vec


def seidel_method(n, m, a, b, c, d, vec, eps, Nmax):
    global iteration_count, final_accuracy, final_residual
    count = len(vec)
    x = np.zeros(count)
    h = (b - a) / n
    k = (d - c) / m
    A = -2 * (1 / h ** 2 + 1 / k ** 2)
    antiH = 1 / h**2
    antiK = 1 / k**2

    for S in range(Nmax):
        x_new = np.copy(x)
        for i in range(count):
            eqi = i % (n - 1) + 1
            eqj = i // (n - 1) + 1
            s1 = 0
            s2 = 0

            if eqi != 1:
                s1 += antiH * x_new[i - 1]

            if eqi != n - 1:
                s2 += antiH * x[i + 1]

            if eqj != 1:
                s1 += antiK * x_new[i - (n - 1)]

            if eqj != m - 1:
                s2 += antiK * x[i + (n - 1)]

            x_new[i] = (vec[i] - s1 - s2) / A
        iteration_count=0
        final_accuracy=0
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            print("Итерации :", S,"Точность на выходе :", np.linalg.norm(x_new - x, ord=np.inf))
            iteration_count = S  # Количество итераций
            final_accuracy = np.linalg.norm(x_new - x, ord=np.inf)
            x = x_new
            break

        x = x_new

    residual = vec - (x * A)
    for i in range(count):
        eqi = i % (n - 1) + 1
        eqj = i // (n - 1) + 1

        if eqi != 1:
            residual[i] -= antiH * x[i - 1]

        if eqi != n - 1:
            residual[i] -= antiH * x[i + 1]

        if eqj != 1:
            residual[i] -= antiK * x[i - (n - 1)]

        if eqj != m - 1:
            residual[i] -= antiK * x[i + (n - 1)]
    final_residual = np.linalg.norm(residual)
    return x, residual


def update_plot():
    a = float(entries['a'].get())
    b = float(entries['b'].get())
    c = float(entries['c'].get())
    d = float(entries['d'].get())
    n = int(entries['n'].get())
    m = int(entries['m'].get())
    eps = float(entries['eps'].get())
    Nmax = int(entries['Nmax'].get())

    vec = buildVector(n, m, a, b, c, d)
    result, residual = seidel_method(n, m, a, b, c, d, vec, eps, Nmax)

    print("Невязка :", np.linalg.norm(residual))

    solution = np.zeros((n + 1, m + 1))
    solution[:, 0] = [mu3(xi, c) for xi in np.linspace(a, b, n + 1)]
    solution[:, -1] = [mu4(xi, d) for xi in np.linspace(a, b, n + 1)]
    solution[0, :] = [mu1(yj, a) for yj in np.linspace(c, d, m + 1)]
    solution[-1, :] = [mu2(yj, b) for yj in np.linspace(c, d, m + 1)]

    for j in range(1, m):
        for i in range(1, n):
            solution[i, j] = result[(j - 1) * (n - 1) + (i - 1)]

    x = np.linspace(a, b, n + 1)
    y = np.linspace(c, d, m + 1)
    X, Y = np.meshgrid(x, y)

    show_table_window(solution, n, m, a, b, c, d)

def show_table_window(solution, n, m, a, b, c, d):
    table_window = tk.Toplevel(root)
    table_window.title("Таблица результатов")

    columns = ('i', 'xi', 'j', 'yj', 'Vij', 'Uij', 'diff')
    table = ttk.Treeview(table_window, columns=columns, show='headings')
    table.heading('i', text='i')
    table.heading('xi', text='xi')
    table.heading('j', text='j')
    table.heading('yj', text='yj')
    table.heading('Vij', text='Vij')
    table.heading('Uij', text='Uij')
    table.heading('diff', text='|Vij - Uij|')

    scrollbar = ttk.Scrollbar(table_window, orient=tk.VERTICAL, command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    h = (b - a) / n
    k = (d - c) / m
    max_diff = 0

    for j in range(m + 1):
        yj = c + j * k
        for i in range(n + 1):
            xi = a + i * h
            Vij = solution[i, j]
            Uij = true_solution(xi, yj)
            diff = abs(Vij - Uij)
            max_diff = max(max_diff, diff)
            table.insert('', tk.END, values=(
                i, "{:.5f}".format(xi), j, "{:.5f}".format(yj), "{:.20f}".format(Vij), "{:.20f}".format(Uij),
                "{:.20f}".format(diff)))

    table.pack(expand=True, fill='both')

    max_diff_label = ctk.CTkLabel(table_window,
                                  text=f"Максимальное отклонение: {max_diff}",
                                  text_color='black')
    max_diff_label.pack()


def show_help():
    help_window = tk.Toplevel(root)
    help_window.title("Справка")
    global iteration_count, final_accuracy, final_residual
    # Текст справки с подстановкой результатов

    help_text = f"""
Программа для численного решения уравнений методом Зейделя.

Параметры ввода:
- a, b, c, d: Границы области решения.
- n, m: Количество разбиений по осям X и Y.
- eps: Точность вычислений.
- Nmax: Максимальное количество итераций метода Зейделя.

Как использовать:
1. Введите параметры в соответствующие поля.
2. Нажмите "Обновить" для вычисления решения и построения графика.
3. Смотрите результаты в графике, таблице и текстовом поле вывода.

Результаты:
- Итерации: {iteration_count}
- Точность на выходе: {final_accuracy:.15f}
- Невязка: {final_residual:.15f}

Обратите внимание: 
- Меньшее значение `eps` обеспечивает более точное решение, но требует больше итераций.
- Максимальное количество итераций ограничивается `Nmax`, чтобы избежать бесконечного цикла.
"""
    text_widget = ctk.CTkTextbox(help_window, height=500, width=500)
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    text_widget.insert(tk.END, help_text)
    text_widget.configure(state="disabled")  # Запрет редактирования текста

root = ctk.CTk()
root.title("Численное решение уравнения")

a_var = tk.StringVar(value="0")
b_var = tk.StringVar(value="1")
c_var = tk.StringVar(value="0")
d_var = tk.StringVar(value="1")
n_var = tk.StringVar(value="3")
m_var = tk.StringVar(value="3")
eps_var = tk.StringVar(value="0.0000001")
Nmax_var = tk.StringVar(value="1000")

entries = {}

frame_boundary_params = ctk.CTkFrame(root)
frame_boundary_params.pack(pady=10, fill=tk.X)

help_button = ctk.CTkButton(root, text="Справка", command=show_help)
help_button.pack(pady=10)

for text, var in zip(['a', 'b', 'c', 'd'], [a_var, b_var, c_var, d_var]):
    label = ctk.CTkLabel(frame_boundary_params, text=text)
    label.pack(side=tk.LEFT, padx=5)
    entries[text] = ctk.CTkEntry(frame_boundary_params, placeholder_text=text, textvariable=var)
    entries[text].pack(side=tk.LEFT, padx=5)

frame_calculation_params = ctk.CTkFrame(root)
frame_calculation_params.pack(pady=10, fill=tk.X)

for text, var in zip(['n', 'm', 'eps', 'Nmax'], [n_var, m_var, eps_var, Nmax_var]):
    label = ctk.CTkLabel(frame_calculation_params, text=text)
    label.pack(side=tk.LEFT, padx=5)
    entries[text] = ctk.CTkEntry(frame_calculation_params, placeholder_text=text, textvariable=var)
    entries[text].pack(side=tk.LEFT, padx=5)

update_button = ctk.CTkButton(root, text="Обновить", command=update_plot)
update_button.pack(pady=10)


root.mainloop()