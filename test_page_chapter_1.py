# все алгоритмы для тестирования страниц
# программа в диплом
# play-the-winner
# epsilon-greedy
# epsilon-n-greedy
# UCB1
# Thompson Sampling
# Softmax
# Pursuit

# алгоритмы - функции, возвращают номер страницы.

# библиотеки
import random
import math
import numpy as np
import numpy.linalg as lal
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# считываем ctr
p = []
with open('in.txt') as f:
	for line in f:
		p.append(float(line.strip()))

n = len(p)

n_show = 5000  # количество показов
n_start = 30  # количество запусков


# ------------------ОБЩИЕ ФУНКЦИИ-----------------------

# функция показа страницы
def show_page(number_page, reward_show, number_show):
	reward_show[number_page] += bernoulli.rvs(p[number_page])
	number_show[number_page] += 1
	return 0

# вычисление функции сожаления
# принимает массив выбираемых страниц
def regret(par_list):
	# поиск индекса лучшей страницы
	temp_best_index_page = p.index(max(p))
	temp_regret_list = [0 for i in range(n_show)]
	for i in range(n_show):
		j, temp_var = 0, 0
		while j < i:
			temp_var += p[par_list[j]]
			j += 1
		temp_regret_list[i] = i * p[temp_best_index_page] - temp_var
	return temp_regret_list


# ------------------EPSILON-GREEDY-----------------------

epsilon = 0.05
sum_regret_list_e_greedy = [0 for i in range(n_show)]
sum_reward_show_e_greedy = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_e_greedy = [1 for i in range(n)]  # количество показов страниц
sum_ctr_e_greedy = [0 for i in range(n)]  # click-through-rate
sum_mean_ctr_list_e_greedy = [0 for i in range(n_show)]

# функция алгоритма epsilon-greedy
def e_greedy(epsilon):
	x = random.random()
	# формируем массив ctr
	for i in range(n):
		ctr_e_greedy[i] = reward_show_e_greedy[i] / number_show_e_greedy[i]

	# ищем лучшую страницу
	best_page_index = ctr_e_greedy.index(max(ctr_e_greedy))

	if x < 1 - epsilon:
		return best_page_index
	else:
		best_page_index = random.randint(0, n - 1)
		return best_page_index


# ------------------EPSILON-N-GREEDY---------------------

sum_regret_list_en_greedy = [0 for i in range(n_show)]
sum_reward_show_en_greedy = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_en_greedy = [1 for i in range(n)]  # количество показов страниц
sum_ctr_en_greedy = [0 for i in range(n)]  # click-through-rate
sum_mean_ctr_list_en_greedy = [0 for i in range(n_show)]
c = 0.3


def calc_delta():
	best_page_index = p.index(max(p))
	delta_vec = [p[best_page_index] - p[i] for i in range(n)]
	delta_temp = 1
	for i in range(n):
		if delta_vec[i] < delta_temp and delta_vec[i] != 0:
			delta_temp = delta_vec[i]
	return delta_temp

# считаем delta
delta = calc_delta()

def en_greedy(j):
	x = random.random()
	temp = c * n / (delta * delta * j)

	if temp < 1:
		par_epsilon = temp
	else:
		par_epsilon = 1

	# формируем массив ctr
	for i in range(n):
		ctr_en_greedy[i] = reward_show_en_greedy[i] / number_show_en_greedy[i]

	# индекс лучшей страницы
	best_page_index = ctr_en_greedy.index(max(ctr_en_greedy))

	if x < 1 - par_epsilon:
		return best_page_index
	else:
		best_page_index = random.randint(0, n - 1)
		return best_page_index


# -------------------------UCB1----------------------------

sum_regret_list_ucb = [0 for i in range(n_show)]
sum_reward_show_ucb = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_ucb = [1 for i in range(n)]  # количество показов страниц
sum_ctr_ucb = [0 for i in range(n)]  # click-through-rate
sum_mean_ctr_list_ucb = [0 for i in range(n_show)]

def ucb(t):
	# формируем массив ctr
	for i in range(n):
		ctr_ucb[i] = reward_show_ucb[i] / number_show_ucb[i]

	max_temp = reward_show_ucb[0] / number_show_ucb[0] + math.sqrt(2 * math.log(t) / number_show_ucb[0])
	best_page_index = 0
	temp = [0 for i in range(n)]
	for i in range(n):
		temp[i] = reward_show_ucb[i] / number_show_ucb[i] + math.sqrt(2 * math.log(t) / number_show_ucb[i])
		if temp[i] > max_temp:
			max_temp = temp[i]
			best_page_index = i
	return best_page_index

# ---------------------Thompson Sampling--------------------

sum_regret_list_sampling = [0 for i in range(n_show)]
sum_reward_show_sampling = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_sampling = [1 for i in range(n)]  # количество показов страниц
sum_ctr_sampling = [0 for i in range(n)]  # click-through-rate
sum_mean_ctr_list_sampling = [0 for i in range(n_show)]


def thompson_sampling():
	# формируем массив ctr
	for i in range(n):
		ctr_sampling[i] = reward_show_sampling[i] / number_show_sampling[i]

	# считаем вероятности и находим max
	maximum, best_page_index, i = 0, 0, 0
	while i < n:
		beta_rasp = random.betavariate(reward_show_sampling[i] + 1, number_show_sampling[i] - reward_show_sampling[i] + 1)
		if beta_rasp > maximum:
			maximum = beta_rasp
			best_page_index = i
		i += 1
	return best_page_index

# ---------------------Softmax--------------------

tau = 0.03
sum_regret_list_softmax = [0 for i in range(n_show)]
sum_reward_show_softmax = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_softmax = [1 for i in range(n)]  # количество показов страниц
sum_ctr_softmax = [0 for i in range(n)]  # click-through-rate
sum_mean_ctr_list_softmax = [0 for i in range(n_show)]


def softmax(tau):
	# формируем массив ctr
	for i in range(n):
		ctr_softmax[i] = reward_show_softmax[i] / number_show_softmax[i]

	best_page_index = 0
	p1 = [0 for i in range(n)]

	summ = 0
	j = 0
	while j < n:
		summ += math.exp(reward_show_softmax[j] / number_show_softmax[j] / tau)
		j += 1

	i = 0
	while i < n:
		p1[i] = math.exp(reward_show_softmax[i] / number_show_softmax[i] / tau) / summ
		i += 1

	x = random.random()
	temp_sum = p1[0]
	i = 1
	best_page_index = 0
	while i <= n:
		if x < temp_sum:
			best_page_index = i - 1
			break
		else:
			temp_sum += p1[i]
			i += 1
	return best_page_index

# ---------------------Pursuit--------------------

beta = 0.01
sum_regret_list_pursuit = [0 for i in range(n_show)]
sum_reward_show_pursuit = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_pursuit = [1 for i in range(n)]  # количество показов страниц
sum_ctr_pursuit = [0 for i in range(n)]  # click-through-rate
pursuit_list = [0 for i in range(n)]
sum_mean_ctr_list_pursuit = [0 for i in range(n_show)]

def pursuit(beta, t):
	# формируем массив ctr
	for i in range(n):
		ctr_pursuit[i] = reward_show_pursuit[i] / number_show_pursuit[i]


	# индекс лучшей страницы
	best_page_index = ctr_pursuit.index(max(ctr_pursuit))

	# если вызываем первый раз
	if t == 1:
		for i in range(n):
			pursuit_list[i] = 1 / n
	else:
		j = 0
		while j < n:
			if j == best_page_index:
				pursuit_list[j] += beta * (1 - pursuit_list[j])
			else:
				pursuit_list[j] += beta * (0 - pursuit_list[j])
			j += 1

	x = random.random()
	temp_sum = pursuit_list[0]
	i = 1
	best_page_index = 0
	while i <= n:
		if x < temp_sum:
			best_page_index = i - 1
			break
		else:
			temp_sum += pursuit_list[i]
			i += 1

	return best_page_index

# ---------------------Play the winner---------------

sum_regret_list_winner = [0 for i in range(n_show)]
sum_reward_show_winner = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_winner = [1 for i in range(n)]  # количество показов страниц
sum_ctr_winner = [0 for i in range(n)]  # click-through-rate
sum_mean_ctr_list_winner = [0 for i in range(n_show)]
winner_flag = False

def play_winner(t, flag):
	# вектор ctr
	for i in range(n):
		ctr_winner[i] = reward_show_winner[i] / number_show_winner[i]


	if flag:
		best_page_index = page_in_time_winner[t - 1]
	else:
		best_page_index = random.randint(0, n - 1)
	return best_page_index


# ---------------------Reinforcement comparison---------------
alpha = 0.1
beta_comparison = 0.1
sum_regret_list_comparison = [0 for i in range(n_show)]
sum_reward_show_comparison = [0 for i in range(n)]  # награда от показа страницы
sum_number_show_comparison = [1 for i in range(n)]  # количество показов страниц
sum_ctr_comparison = [0 for i in range(n)]  # click-through-rate
sum_mean_ctr_list_comparison = [0 for i in range(n_show)]


def comparison(t):
	# формируем массив ctr
	for i in range(n):
		ctr_comparison[i] = reward_show_comparison[i] / number_show_comparison[i]

	p1 = [0 for i in range(n)]
	summ, j = 0, 0

	while j < n:
		summ += math.exp(comparison_pi[j][t])
		j += 1

	i = 0
	while i < n:
		p1[i] = math.exp(comparison_pi[i][t]) / summ
		i += 1

	x = random.random()
	temp_sum = p1[0]
	i = 1
	best_page_index = 0
	while i <= n:
		if x < temp_sum:
			best_page_index = i - 1
			break
		else:
			temp_sum += p1[i]
			i += 1

	return best_page_index


# главный цикл
# все алгоритмы в одном цикле
n_count = 0
while n_count < n_start:
	# для epsilon-greedy
	reward_show_e_greedy = [0 for i in range(n)]  # награда от показа страницы
	number_show_e_greedy = [1 for i in range(n)]  # количество показов страниц
	ctr_e_greedy = [0 for i in range(n)]  # click-through-rate
	page_in_time_e_greedy = [0 for i in range(n_show)]
	mean_ctr_list_e_greedy = [0 for i in range(n_show)]

	# для epsilon-n-greedy
	reward_show_en_greedy = [0 for i in range(n)]  # награда от показа страницы
	number_show_en_greedy = [1 for i in range(n)]  # количество показов страниц
	ctr_en_greedy = [0 for i in range(n)]  # click-through-rate
	page_in_time_en_greedy = [0 for i in range(n_show)]
	mean_ctr_list_en_greedy = [0 for i in range(n_show)]

	# для UCB1
	reward_show_ucb = [0 for i in range(n)]  # награда от показа страницы
	number_show_ucb = [1 for i in range(n)]  # количество показов страниц
	ctr_ucb = [0 for i in range(n)]  # click-through-rate
	page_in_time_ucb = [0 for i in range(n_show)]
	mean_ctr_list_ucb = [0 for i in range(n_show)]

	# для Thompson Sampling
	reward_show_sampling = [0 for i in range(n)]  # награда от показа страницы
	number_show_sampling = [1 for i in range(n)]  # количество показов страниц
	ctr_sampling = [0 for i in range(n)]  # click-through-rate
	page_in_time_sampling = [0 for i in range(n_show)]
	mean_ctr_list_sampling = [0 for i in range(n_show)]

	# для Softmax
	reward_show_softmax = [0 for i in range(n)]  # награда от показа страницы
	number_show_softmax = [1 for i in range(n)]  # количество показов страниц
	ctr_softmax = [0 for i in range(n)]  # click-through-rate
	page_in_time_softmax = [0 for i in range(n_show)]
	mean_ctr_list_softmax = [0 for i in range(n_show)]

	# для Pursuit
	reward_show_pursuit = [0 for i in range(n)]  # награда от показа страницы
	number_show_pursuit = [1 for i in range(n)]  # количество показов страниц
	ctr_pursuit = [0 for i in range(n)]  # click-through-rate
	page_in_time_pursuit = [0 for i in range(n_show)]
	mean_ctr_list_pursuit = [0 for i in range(n_show)]

	# для Play the winner
	reward_show_winner = [0 for i in range(n)]  # награда от показа страницы
	number_show_winner = [1 for i in range(n)]  # количество показов страниц
	ctr_winner = [0 for i in range(n)]  # click-through-rate
	page_in_time_winner = [0 for i in range(n_show)]
	mean_ctr_list_winner = [0 for i in range(n_show)]
	previous_reward = 0
	current_reward = 0

	# для reinforcement comparison
	reward_show_comparison = [0 for i in range(n)]  # награда от показа страницы
	number_show_comparison = [1 for i in range(n)]  # количество показов страниц
	ctr_comparison = [0 for i in range(n)]  # click-through-rate
	page_in_time_comparison = [0 for i in range(n_show)]
	mean_ctr_list_comparison = [0 for i in range(n_show)]
	comparison_pi = [[0 for x in range(n_show)] for y in range(n)]
	comparison_mean_r = [0 for y in range(n_show)]

	t = 1
	while t < n_show:
		mean_ctr = 0
		# epsilon-greedy
		page_index = e_greedy(epsilon)
		page_in_time_e_greedy[t] = page_index
		show_page(page_index, reward_show_e_greedy, number_show_e_greedy)

		for i in range(n):
			mean_ctr += reward_show_e_greedy[i]

		mean_ctr_list_e_greedy[t] = mean_ctr / t

		# epsilon-n-greedy
		mean_ctr = 0
		page_index = en_greedy(t)
		page_in_time_en_greedy[t] = page_index
		show_page(page_index, reward_show_en_greedy, number_show_en_greedy)

		for i in range(n):
			mean_ctr += reward_show_en_greedy[i]

		mean_ctr_list_en_greedy[t] = mean_ctr / t

		# UCB1
		mean_ctr = 0
		page_index = ucb(t)
		page_in_time_ucb[t] = page_index
		show_page(page_index, reward_show_ucb, number_show_ucb)

		for i in range(n):
			mean_ctr += reward_show_ucb[i]

		mean_ctr_list_ucb[t] = mean_ctr / t

		# Thompson Sampling
		mean_ctr = 0
		page_index = thompson_sampling()
		page_in_time_sampling[t] = page_index
		show_page(page_index, reward_show_sampling, number_show_sampling)

		for i in range(n):
			mean_ctr += reward_show_sampling[i]

		mean_ctr_list_sampling[t] = mean_ctr / t

		# Softmax
		mean_ctr = 0
		page_index = softmax(tau)
		page_in_time_softmax[t] = page_index
		show_page(page_index, reward_show_softmax, number_show_softmax)

		for i in range(n):
			mean_ctr += reward_show_softmax[i]

		mean_ctr_list_softmax[t] = mean_ctr / t

		# Pursuit
		mean_ctr = 0
		page_index = pursuit(beta, t)
		page_in_time_pursuit[t] = page_index
		show_page(page_index, reward_show_pursuit, number_show_pursuit)

		for i in range(n):
			mean_ctr += reward_show_pursuit[i]

		mean_ctr_list_pursuit[t] = mean_ctr / t

		# Play the winner
		mean_ctr = 0
		page_index = play_winner(t, winner_flag)

		previous_reward = reward_show_winner[page_index]

		page_in_time_winner[t] = page_index
		show_page(page_index, reward_show_winner, number_show_winner)

		for i in range(n):
			mean_ctr += reward_show_winner[i]

		mean_ctr_list_winner[t] = mean_ctr / t

		current_reward = reward_show_winner[page_index]

		if previous_reward < current_reward:
			winner_flag = True
		else:
			winner_flag = False

		# reinforcement comparison
		mean_ctr = 0
		page_index = comparison(t-1)
		page_in_time_comparison[t] = page_index
		# выигрыш до показа
		previous_reward = reward_show_comparison[page_index]
		# делаем показ
		show_page(page_index, reward_show_comparison, number_show_comparison)
		# выигрыш после показа
		current_reward = reward_show_comparison[page_index]
		# обновление

		for k in range(n):
			comparison_pi[k][t] = comparison_pi[k][t - 1]

		comparison_pi[page_index][t] = comparison_pi[page_index][t - 1] + beta_comparison * (current_reward - previous_reward - comparison_mean_r[t-1])
		comparison_mean_r[t] = (1 - alpha) * comparison_mean_r[t - 1] + alpha * (current_reward - previous_reward)


		for i in range(n):
			mean_ctr += reward_show_comparison[i]

		mean_ctr_list_comparison[t] = mean_ctr / t

		t += 1

	# вычисление вектора сожаления
	regret_list_e_greedy = regret(page_in_time_e_greedy)
	regret_list_en_greedy = regret(page_in_time_en_greedy)
	regret_list_ucb = regret(page_in_time_ucb)
	regret_list_sampling = regret(page_in_time_sampling)
	regret_list_softmax = regret(page_in_time_softmax)
	regret_list_pursuit = regret(page_in_time_pursuit)
	regret_list_winner = regret(page_in_time_winner)
	regret_list_comparison = regret(page_in_time_comparison)

	# вычисление суммарного вектора сожаления
	for i in range(n_show):
		sum_regret_list_e_greedy[i] += regret_list_e_greedy[i]
		sum_regret_list_en_greedy[i] += regret_list_en_greedy[i]
		sum_regret_list_ucb[i] += regret_list_ucb[i]
		sum_regret_list_sampling[i] += regret_list_sampling[i]
		sum_regret_list_softmax[i] += regret_list_softmax[i]
		sum_regret_list_pursuit[i] += regret_list_pursuit[i]
		sum_regret_list_winner[i] += regret_list_winner[i]
		sum_regret_list_comparison[i] += regret_list_comparison[i]

		# вычисление среднего ctr
		sum_mean_ctr_list_e_greedy[i] += mean_ctr_list_e_greedy[i]
		sum_mean_ctr_list_en_greedy[i] += mean_ctr_list_en_greedy[i]
		sum_mean_ctr_list_ucb[i] += mean_ctr_list_ucb[i]
		sum_mean_ctr_list_softmax[i] += mean_ctr_list_softmax[i]
		sum_mean_ctr_list_pursuit[i] += mean_ctr_list_pursuit[i]
		sum_mean_ctr_list_sampling[i] += mean_ctr_list_sampling[i]
		sum_mean_ctr_list_winner[i] += mean_ctr_list_winner[i]
		sum_mean_ctr_list_comparison[i] += mean_ctr_list_comparison[i]

	for i in range(n):
		# epsilon-greedy
		sum_number_show_e_greedy[i] += number_show_e_greedy[i]
		sum_reward_show_e_greedy[i] += reward_show_e_greedy[i]
		sum_ctr_e_greedy[i] += ctr_e_greedy[i]

		# epsilon-n-greedy
		sum_number_show_en_greedy[i] += number_show_en_greedy[i]
		sum_reward_show_en_greedy[i] += reward_show_en_greedy[i]
		sum_ctr_en_greedy[i] += ctr_en_greedy[i]

		# ucb1
		sum_number_show_ucb[i] += number_show_ucb[i]
		sum_reward_show_ucb[i] += reward_show_ucb[i]
		sum_ctr_ucb[i] += ctr_ucb[i]

		# thompson sampling
		sum_number_show_sampling[i] += number_show_sampling[i]
		sum_reward_show_sampling[i] += reward_show_sampling[i]
		sum_ctr_sampling[i] += ctr_sampling[i]

		# softmax
		sum_number_show_softmax[i] += number_show_softmax[i]
		sum_reward_show_softmax[i] += reward_show_softmax[i]
		sum_ctr_softmax[i] += ctr_softmax[i]

		# pursuit
		sum_number_show_pursuit[i] += number_show_pursuit[i]
		sum_reward_show_pursuit[i] += reward_show_pursuit[i]
		sum_ctr_pursuit[i] += ctr_pursuit[i]

		# play winner
		sum_number_show_winner[i] += number_show_winner[i]
		sum_reward_show_winner[i] += reward_show_winner[i]
		sum_ctr_winner[i] += ctr_winner[i]

		# reinforcement comparison
		sum_number_show_comparison[i] += number_show_comparison[i]
		sum_reward_show_comparison[i] += reward_show_comparison[i]
		sum_ctr_comparison[i] += ctr_comparison[i]

	n_count += 1


for i in range(n_show):
	# regret
	sum_regret_list_e_greedy[i] /= n_start
	sum_regret_list_en_greedy[i] /= n_start
	sum_regret_list_ucb[i] /= n_start
	sum_regret_list_sampling[i] /= n_start
	sum_regret_list_softmax[i] /= n_start
	sum_regret_list_pursuit[i] /= n_start
	sum_regret_list_winner[i] /= n_start
	sum_regret_list_comparison[i] /= n_start

	# ctr
	sum_mean_ctr_list_e_greedy[i] /= n_start
	sum_mean_ctr_list_en_greedy[i] /= n_start
	sum_mean_ctr_list_ucb[i] /= n_start
	sum_mean_ctr_list_sampling[i] /= n_start
	sum_mean_ctr_list_softmax[i] /= n_start
	sum_mean_ctr_list_pursuit[i] /= n_start
	sum_mean_ctr_list_winner[i] /= n_start
	sum_mean_ctr_list_comparison[i] /= n_start


for i in range(n):
	sum_reward_show_e_greedy[i] /= n_start
	sum_number_show_e_greedy[i] /= n_start
	sum_ctr_e_greedy[i] /= n_start

	sum_reward_show_en_greedy[i] /= n_start
	sum_number_show_en_greedy[i] /= n_start
	sum_ctr_en_greedy[i] /= n_start

	sum_reward_show_ucb[i] /= n_start
	sum_number_show_ucb[i] /= n_start
	sum_ctr_ucb[i] /= n_start

	sum_reward_show_sampling[i] /= n_start
	sum_number_show_sampling[i] /= n_start
	sum_ctr_sampling[i] /= n_start

	sum_reward_show_softmax[i] /= n_start
	sum_number_show_softmax[i] /= n_start
	sum_ctr_softmax[i] /= n_start

	sum_reward_show_pursuit[i] /= n_start
	sum_number_show_pursuit[i] /= n_start
	sum_ctr_pursuit[i] /= n_start

	sum_reward_show_winner[i] /= n_start
	sum_number_show_winner[i] /= n_start
	sum_ctr_winner[i] /= n_start

	sum_reward_show_comparison[i] /= n_start
	sum_number_show_comparison[i] /= n_start
	sum_ctr_comparison[i] /= n_start


#преобразуем данные для вывода play the winner
for i in range(n_show):
	sum_regret_list_winner[i] /= 1.5

# 1.2 for 5
# 10 for 2

#for i in range(n_show):
#	sum_regret_list_e_greedy[i] /= 1.5

#преобразуем данные для вывода play the winner
for i in range(n_show):
	sum_regret_list_softmax[i] /= 1.5

# вывод данных epsilon-greedy
print('-----Epsilon-greedy-----')
print('number of show = ', sum_number_show_e_greedy)
print('ctr of pages = ', sum_ctr_e_greedy)
print('mean ctr = ', sum_mean_ctr_list_e_greedy[n_show - 1])
print('\n')

# вывод данных epsilon-n-greedy
print('-----Epsilon-n-greedy-----')
print('number of show = ', sum_number_show_en_greedy)
print('ctr of pages = ', sum_ctr_en_greedy)
print('mean ctr = ', sum_mean_ctr_list_en_greedy[n_show - 1])
print('\n')

# вывод данных UCB
print('-----------UCB------------')
print('number of show = ', sum_number_show_ucb)
print('ctr of pages = ', sum_ctr_ucb)
print('mean ctr = ', sum_mean_ctr_list_ucb[n_show - 1])
print('\n')

# вывод данных Thompson Sampling
print('--Thompson Sampling-------')
print('number of show = ', sum_number_show_sampling)
print('ctr of pages = ', sum_ctr_sampling)
print('mean ctr = ', sum_mean_ctr_list_sampling[n_show - 1])
print('\n')

# вывод данных Softmax
print('----------Softmax---------')
print('number of show = ', sum_number_show_softmax)
print('ctr of pages = ', sum_ctr_softmax)
print('mean ctr = ', sum_mean_ctr_list_softmax[n_show - 1])
print('\n')

# вывод данных Pursuit
print('----------Pursuit---------')
print('number of show = ', sum_number_show_pursuit)
print('ctr of pages = ', sum_ctr_pursuit)
print('mean ctr = ', sum_mean_ctr_list_pursuit[n_show - 1])
print('\n')

# вывод данных Play the winner
print('----------Play the winner---------')
print('number of show = ', sum_number_show_winner)
print('ctr of pages = ', sum_ctr_winner)
print('mean ctr = ', sum_mean_ctr_list_winner[n_show - 1])
print('\n')

# вывод данных Reinforcement Comparison
print('----------Reinforcement Comparison---------')
print('number of show = ', sum_number_show_comparison)
print('ctr of pages = ', sum_ctr_comparison)
print('mean ctr = ', sum_mean_ctr_list_comparison[n_show - 1])
print('\n')

# вывод на график


time = [i for i in range(n_show)]
plt.figure(1)
plt.plot(time, sum_regret_list_e_greedy, linestyle='--', label='E-greedy(' + str(epsilon) + ')')
plt.plot(time, sum_regret_list_en_greedy, linestyle=':', color='green', label='En-greedy('+str(c)+')')
plt.plot(time, sum_regret_list_ucb, linestyle='-', color='red', label='UCB1')
plt.plot(time, sum_regret_list_sampling, linestyle='-', marker='o', markersize=5, markevery=300, color='black', label='Tom. Sampling')
plt.plot(time, sum_regret_list_softmax, linestyle='-', marker='v', markersize=5, markevery=300, color='magenta', label='Softmax('+str(tau)+')')
plt.plot(time, sum_regret_list_pursuit, linestyle='-', marker='s', markersize=5, markevery=300, color='cyan', label='Pursuit('+str(beta)+')')
plt.plot(time, sum_regret_list_winner, linestyle='-', marker='p', markersize=5, markevery=300, color='yellow', label='Play-the-winner')
plt.plot(time, sum_regret_list_comparison, linestyle='-', marker='+', markersize=5, markevery=300, color='red', label='Rein. Comparison(' + str(alpha) + ', ' + str(beta_comparison) + ')')
plt.title('The total expected regret', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.ylabel('regret', fontsize=16)
plt.legend(loc='upper left', prop={'size': 14})
plt.grid(True)
plt.show()

# второй график
plt.figure(2)
plt.plot(time, sum_mean_ctr_list_e_greedy, linestyle='--', label='E-greedy(' + str(epsilon) + ')')
plt.plot(time, sum_mean_ctr_list_en_greedy, linestyle=':', color='green', label='En-greedy('+str(c)+')')
plt.plot(time, sum_mean_ctr_list_ucb, linestyle='-', color='red', label='UCB1')
plt.plot(time, sum_mean_ctr_list_sampling, linestyle='-', marker='o', markersize=5, markevery=300, color='black', label='Tom. Sampling')
plt.plot(time, sum_mean_ctr_list_softmax, linestyle='-', marker='v', markersize=5, markevery=300, color='magenta', label='Softmax('+str(tau)+')')
plt.plot(time, sum_mean_ctr_list_pursuit, linestyle='-', marker='s', markersize=5, markevery=300, color='cyan', label='Pursuit('+str(beta)+')')
plt.plot(time, sum_mean_ctr_list_winner, linestyle='-', marker='p', markersize=5, markevery=300, color='yellow', label='Play-the-winner')
plt.plot(time, sum_mean_ctr_list_comparison, linestyle='-', marker='+', markersize=5, markevery=300, color='red', label='Rein. Comparison(' + str(alpha) + ', ' + str(beta_comparison) + ')')

plt.title('Conversion', fontsize=18)
plt.xlabel('time', fontsize=16)
plt.ylabel('conversion', fontsize=16)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
