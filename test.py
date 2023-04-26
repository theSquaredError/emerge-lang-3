import numpy as np



def check_game_over(markers):
	global game_over
	global winner

	x_pos = 0
	for x in markers:
		#check columns
		if sum(x) == 3:
			winner = 1
			game_over = True
		if sum(x) == -3:
			winner = 2
			game_over = True
		
		#check rows
		if markers[0][x_pos] + markers [1][x_pos] + markers [2][x_pos] == 3:
			winner = 1
			game_over = True
		if markers[0][x_pos] + markers [1][x_pos] + markers [2][x_pos] == -3:
			winner = 2
			game_over = True
		x_pos += 1

	#check cross
	if markers[0][0] + markers[1][1] + markers [2][2] == 3 or markers[2][0] + markers[1][1] + markers [0][2] == 3:
		winner = 1
		game_over = True
	if markers[0][0] + markers[1][1] + markers [2][2] == -3 or markers[2][0] + markers[1][1] + markers [0][2] == -3:
		winner = 2
		game_over = True

	#check for tie
	if game_over == False:
		tie = True
		for row in markers:
			for i in row:
				if i == 0:
					tie = False
		#if it is a tie, then call game over and set winner to 0 (no one)
		if tie == True:
			game_over = True
			winner = 0


def check_game_over_2(x):
		y = x.copy()
		y = np.array(y).reshape(3,3)
		y[np.where(y==2)] = -1
		# check for game over
		game_over = False
		check_game_over(y)







states_0 = [[0]*9]

states_1 = []
board_pos = [i for i in range(9)]
for i in board_pos:
  for j in board_pos:
    if i!=j:
      x = [0]*9
      x[i] = 1
      x[j] = 2
      states_1.append(x)

# states_2
states_2 = []
for s in states_1:
  indices = [i for i, x in enumerate(s) if x == 1 or x == 2]
  board_pos = [i for i in range(9)]
  board_pos = list(set(board_pos) - set(indices))
  for i in board_pos:
    for j in board_pos:
      if i!=j:
        x = s.copy()
        x[i] = 1
        x[j] = 2
        if x not in states_2:
          states_2.append(x)

# states_3
TERMINAL_STATES = []
states_3 = []
for s in states_2:
  indices = [i for i, x in enumerate(s) if x == 1 or x == 2]
  board_pos = [i for i in range(9)]
  board_pos = list(set(board_pos) - set(indices))
  # print(len(board_pos))
  for i in board_pos:
    for j in board_pos:
      if i!=j:
        x = s.copy()
        # print(x,i,j)
        x[i] = 1
        # y = x.copy()
        # y = np.array(y).reshape(3,3)
        # y[np.where(y==2)] = -1
        # # check for game over
        game_over = False
        check_game_over_2(x)
        # if game is over then state_3.append(x) else
        if(game_over == True):
          # print("matrix\n",y)
          if x not in TERMINAL_STATES:
            TERMINAL_STATES.append(x)
            break
        else:
          x[j] = 2
          if x not in states_3:
            states_3.append(x)

# states_4
states_4 = []
# check for the
for s in states_3:
  indices = [i for i, x in enumerate(s) if x == 1 or x == 2]
  board_pos = [i for i in range(9)]
  board_pos = list(set(board_pos) - set(indices))
  for i in board_pos:
    for j in board_pos:
      if i!=j:
        x = s.copy()
        x[i] = 1
        x[j] = 2
        if x not in states_4:
          states_4.append(x)
STATES = states_0+states_1+states_2+states_3+states_4+TERMINAL_STATES
# print(f'len(states_0) = {len(states_0)}, len(states_1) = {len(states_1)}, len(states_2) = {len(states_2)}, len(states_3) = {len(states_3)}, len(states_4) = {len(states_4)}')
print(f'len(STATES) = {len(TERMINAL_STATES)}')