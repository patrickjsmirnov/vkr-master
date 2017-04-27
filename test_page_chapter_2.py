import math
import numpy as np
import numpy.linalg as lal
import matplotlib.pyplot as plt
import random
from scipy.stats import bernoulli

p_context = [0.5, 0.5]

p = []
m = 0

# считываем вероятности
def read(f):
	global m
	m = 0
	for line in f:
		m += 1
		row = [float(i) for i in line.split()]
		p.append(row)
	return 0


f = open('contextual_in.txt', 'r')
read(f)

n = len(p[0])
print(p)
print('n = ', n)  # количество страниц
print('m = ', m)  # количество объявлений

t = 5000  # количество показов
launch_number = 30  # количество запусков


alpha = 3  # параметр алгоритма

# генерируем контекст
# возвращает номер объявления
def context():
	x = random.random()
	# print(x)
	temp_sum = p_context[0]
	i = 1
	number_of_ad = 0
	while i <= m:
		# print('temp_sum = ', temp_sum)
		if x < temp_sum:
			number_of_ad = i - 1
			break
		else:
			temp_sum += p_context[i]
			i += 1
	# print(number_of_ad)
	return number_of_ad



# показываем страницу
def show_page(index_page):
	# p_main - текущий вектор конверсий объявлений
	current_reward = bernoulli.rvs(p_main[index_page])
	reward_show_main[index_page] += current_reward
	number_show_main[index_page] += 1

	# обновляем данные объявления
	reward_show[ad_number][index_page] += current_reward
	number_show[ad_number][index_page] += 1

	return current_reward


# вычисление функции сожаления
def func_regret_calculation(par_list1, par_list2):
	for i in range(t):
		j, temp_var_best, temp_var_current = 0, 0, 0
		while j < i:
			temp_var_best += par_list1[j]
			temp_var_current += par_list2[j]
			j += 1
		regret_list[i] = temp_var_best - temp_var_current
	return 0


a = [0 for i in range(n)]
b = [0 for i in range(n)]
teta = [0 for i in range(n)]

sum_conversion_page = [[0 for i in range(n)] for i in range(m)]  # суммарные конверсии страниц по объявлениям
sum_regret_list = [0 for i in range(t)]
sum_mean_conversion_list = [0 for i in range(t)]
sum_number_context = [0 for i in range(m)]
sum_number_show = [[0 for i in range(n)] for i in range(m)]

x = [[0 for i in range(1)] for j in range(m)]  # context vector
p_alg = [0 for i in range(n)]


launch_count = 0
while launch_count < launch_number:
	number_show = [[1 for i in range(n)] for i in range(m)]  # показы
	reward_show = [[0 for i in range(n)] for i in range(m)]  # выигрыши
	reward_show_main = [0 for i in range(n)]
	number_show_main = [0 for i in range(n)]
	conversion_page = [[0 for i in range(n)] for i in range(m)]  # конверсии страниц по объявлениям
	best_conversion = [0 for i in range(t)]
	current_conversion = [0 for i in range(t)]
	regret_list = [0 for i in range(t)]
	mean_conversion_list = [0 for i in range(t)]
	number_context = [0 for i in range(m)]
	p_main = [0 for i in range(n)]

	# создание матриц
	for i in range(n):
		a[i] = np.eye(m)  # единичные матрицы
		b[i] = np.matrix([[0 for j in range(1)] for k in range(m)])
		teta[i] = np.matrix([[0 for j in range(1)] for k in range(m)])

	time = 1
	while time < t:
		# генерируем номер объявления
		ad_number = context()
		number_context[ad_number] += 1
		# изменяем контекст
		x = np.matrix([[0 for i in range(1)] for j in range(m)])
		x[ad_number][0] = 1

		# выбрать текущий вектор это p_main
		for i in range(n):
			p_main[i] = p[ad_number][i]

		max_p = 0
		max_i = 0
		mean_conversion = 0

		# лучшая текущая текущая конверсия
		best_conversion[time] = max(p_main)

		for i in range(n):
			teta[i] = lal.linalg.inv(a[i]) * b[i]
			p_alg[i] = teta[i].transpose() * x + alpha * math.sqrt(x.transpose() * lal.linalg.inv(a[i]) * x)
			if p_alg[i] > max_p:
				max_p = p_alg[i]
				max_i = i

		# текущая конверсия
		current_conversion[time] = p_main[max_i]

		# показываем выбранную страницу
		reward = show_page(max_i)

		# обновляем данные
		a[max_i] += x * x.transpose()
		b[max_i] += reward * x

		# conversion
		for i in range(n):
			mean_conversion += reward_show_main[i]

		mean_conversion_list[time] = mean_conversion / time

		time += 1

	func_regret_calculation(best_conversion, current_conversion)

	for k in range(t):
		sum_regret_list[k] += regret_list[k]
		sum_mean_conversion_list[k] += mean_conversion_list[k]

	for i in range(m):
		sum_number_context[i] += number_context[i]
		for j in range(n):
			conversion_page[i][j] = reward_show[i][j] / number_show[i][j]
			sum_conversion_page[i][j] += conversion_page[i][j]
			sum_number_show[i][j] += number_show[i][j]

	launch_count += 1


for i in range(t):
	sum_regret_list[i] /= launch_number
	sum_mean_conversion_list[i] /= launch_number

for i in range(m):
	sum_number_context[i] /= launch_number
	for j in range(n):
		sum_conversion_page[i][j] /= launch_number
		sum_number_show[i][j] /= launch_number


# выбор страницы с наибольшей конверсией
mean_of_conversion = [0 for i in range(n)]
for i in range(n):
	temp_sum = 0
	for j in range(m):
		temp_sum += sum_number_context[j] * sum_conversion_page[j][i]
	mean_of_conversion[i] = temp_sum


print('all conv = ', mean_of_conversion)

print('best page is ', mean_of_conversion.index(max(mean_of_conversion)))

print('conversion of page = ', sum_conversion_page)
print('\n')

print('number of show = ', sum_number_show)
print('\n')

print('number of context = ', sum_number_context)
print('\n')

print('summary conversion = ', sum_mean_conversion_list[t - 1])


time_list = [i for i in range(t)]
plt.figure(1)
plt.plot(time_list, sum_regret_list, linestyle='--', label='LinUCB')
plt.title('LinUCB', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.ylabel('regret', fontsize=16)
plt.legend(loc='upper left', prop={'size': 14})
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(time_list, sum_mean_conversion_list, linestyle='--', color='blue', label='LinUCB')
plt.title('LinUCB', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.ylabel('conversion', fontsize=16)
plt.legend(loc='lower right', prop={'size': 14})
plt.grid(True)
plt.show()
