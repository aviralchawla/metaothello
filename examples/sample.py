from metaothello.games import ClassicOthello, DeleteFlanking, NoMiddleFlip

co = ClassicOthello()
co.print_board()
co.get_all_valid_moves()

for _ in range(60):
    move = co.get_random_valid_move()
    if move is None:
        break
    co.play_move(move)
    co.print_board()

print(co.history)

co = ClassicOthello()
co.generate_random_game()
print(co.history)

nmf = NoMiddleFlip()
nmf.print_board()
nmf.get_all_valid_moves()

for _ in range(60):
    move = nmf.get_random_valid_move()
    if move is None:
        break
    nmf.play_move(move)
    nmf.print_board()

print(nmf.history)

df = DeleteFlanking()
df.print_board()
df.get_all_valid_moves()

for _ in range(60):
    move = df.get_random_valid_move()
    if move is None:
        break
    df.play_move(move)
    df.print_board()

print(df.history)
