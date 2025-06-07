import sys, os
import math
from collections import deque
from itertools import product
from tqdm import tqdm

sys.setrecursionlimit(10**7)
input = sys.stdin.readline
clear_cmd = "cls" if os.name == "nt" else "clear"

# ////------------------ 엔드게임 ------------------////

id_map = {}
state_n = 0

ret = []
degree = []
max_child = []
sidea = []
preds = []

king = [-1, -1, -1, 0, 0, 1, 1, 1]
king1 = [-1, 0, 1, -1, 1, -1, 0, 1]

def query():
    print("> ", end='', flush=True)

def rank_of(p):
    return p >> 3

def file_of(p):
    return p & 7

def inb(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def kings_adj(a, b):
    ra, ca = rank_of(a), file_of(a)
    rb, cb = rank_of(b), file_of(b)
    return max(abs(ra - rb), abs(ca - cb)) <= 1

def attacked_by_white(sq, wp, wkt, wk):
    r_sq, c_sq = rank_of(sq), file_of(sq)
    r_wk, c_wk = rank_of(wk), file_of(wk)

    if max(abs(r_sq - r_wk), abs(c_sq - c_wk)) == 1:
        return True

    r_wp, c_wp = rank_of(wp), file_of(wp)

    if wkt == 0:
        if r_wp - 1 == r_sq and (c_wp - 1 == c_sq or c_wp + 1 == c_sq):
            return True
        return False

    if wkt in (2, 1):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r_wp + dr, c_wp + dc
            while inb(rr, cc):
                sq2 = rr * 8 + cc
                if sq2 == sq:
                    return True
                if sq2 == wk:
                    break
                rr += dr
                cc += dc

    if wkt == 1:
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rr, cc = r_wp + dr, c_wp + dc
            while inb(rr, cc):
                sq2 = rr * 8 + cc
                if sq2 == sq:
                    return True
                if sq2 == wk:
                    break
                rr += dr
                cc += dc

    return False

def preprocess():
    global id_map, state_n, ret, degree, max_child, sidea, preds

    print("전처리 중입니다...")
    total_combinations = 2**19 * 3

    for wkt, wp, wk, bk, side in tqdm(
        product(range(3), range(64), range(64), range(64), range(2)),
        total=total_combinations,
        desc="맵 배열 빌드",
        ncols=80,
    ):
        r_wp = rank_of(wp)

        if wkt == 0 and (r_wp == 0 or r_wp == 7):
            continue

        if wk == wp or bk == wp or bk == wk:
            continue

        if kings_adj(wk, bk):
            continue
        id_map[(wkt, wp, wk, bk, side)] = state_n
        state_n += 1

    ret = [-1] * state_n
    degree = [0] * state_n
    max_child = [-1] * state_n
    sidea = [False] * state_n
    preds = [[] for _ in range(state_n)]

    items = list(id_map.items())

    for (wkt, wp, wk, bk, side), sid in tqdm(
        items, total=state_n, desc="상태 초기화", ncols=80
    ):
        sidea[sid] = side == 1
        isBlack = side == 1

        r_wp, c_wp = rank_of(wp), file_of(wp)
        r_wk, c_wk = rank_of(wk), file_of(wk)
        r_bk, c_bk = rank_of(bk), file_of(bk)

        if not isBlack:

            for d in range(8):
                nr = r_wk + king[d]
                nc = c_wk + king1[d]
                if not inb(nr, nc):
                    continue
                nt = nr * 8 + nc
                if nt == wp or nt == bk:
                    continue
                if kings_adj(nt, bk):
                    continue
                nid = id_map.get((wkt, wp, nt, bk, 1), -1)
                if nid >= 0:
                    degree[sid] += 1
                    preds[nid].append(sid)

            if wkt == 0:

                fwd1 = wp - 8
                if 0 <= fwd1 < 64 and fwd1 != wk and fwd1 != bk:
                    rnf = rank_of(fwd1)
                    if rnf == 0:

                        for newt in (1, 2):
                            nid = id_map.get((newt, fwd1, wk, bk, 1), -1)
                            if nid >= 0:
                                degree[sid] += 1
                                preds[nid].append(sid)
                    else:
                        nid = id_map.get((0, fwd1, wk, bk, 1), -1)
                        if nid >= 0:
                            degree[sid] += 1
                            preds[nid].append(sid)

                if r_wp == 6:
                    fwd2 = wp - 16
                    if 0 <= fwd2 < 64:
                        between = wp - 8
                        if (
                            between != wk
                            and between != bk
                            and fwd2 != wk
                            and fwd2 != bk
                        ):
                            nid = id_map.get((0, fwd2, wk, bk, 1), -1)
                            if nid >= 0:
                                degree[sid] += 1
                                preds[nid].append(sid)

                for dc in (-1, 1):
                    nf = c_wp + dc
                    nr = r_wp - 1
                    if 0 <= nf < 8 and nr >= 0:
                        tgt = nr * 8 + nf
                        if tgt == bk:

                            if ret[sid] < 0:
                                ret[sid] = 0
                            continue
                        nid = id_map.get((wkt, tgt, wk, bk, 1), -1)
                        if nid >= 0:
                            degree[sid] += 1
                            preds[nid].append(sid)
            else:

                rook_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                bishop_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                dirs = []
                if wkt in (2, 1):
                    dirs.extend(rook_dirs)
                if wkt == 1:
                    dirs.extend(bishop_dirs)

                for dr, dc in dirs:
                    rr, cc = r_wp + dr, c_wp + dc
                    while inb(rr, cc):
                        tgt = rr * 8 + cc
                        if tgt == wk:
                            break
                        if tgt == bk:

                            if ret[sid] < 0:
                                ret[sid] = 0
                            break
                        nid = id_map.get((wkt, tgt, wk, bk, 1), -1)
                        if nid >= 0:
                            degree[sid] += 1
                            preds[nid].append(sid)
                        rr += dr
                        cc += dc

        else:
            for d in range(8):
                nr = r_bk + king[d]
                nc = c_bk + king1[d]
                if not inb(nr, nc):
                    continue
                nt = nr * 8 + nc

                if nt == wk:
                    continue
                if kings_adj(nt, wk):
                    continue
                if attacked_by_white(nt, wp, wkt, wk):
                    continue

                captures = nt == wp
                degree[sid] += 1
                if captures:
                    continue
                nid = id_map.get((wkt, wp, wk, nt, 0), -1)
                if nid >= 0:
                    preds[nid].append(sid)

            if degree[sid] == 0 and attacked_by_white(bk, wp, wkt, wk):
                if ret[sid] < 0:
                    ret[sid] = 0

    print("전처리가 완료되었습니다!")

    queue = deque()
    for sid in range(state_n):
        if ret[sid] == 0:
            queue.append(sid)

    while queue:
        u = queue.popleft()
        d_u = ret[u]
        for v in preds[u]:
            if ret[v] >= 0:
                continue
            if not sidea[v]:
                ret[v] = d_u + 1
                queue.append(v)
            else:
                degree[v] -= 1
                if d_u > max_child[v]:
                    max_child[v] = d_u
                if degree[v] == 0:
                    ret[v] = max_child[v] + 1
                    queue.append(v)

def KPvK():
    board = [None] * 8
    for i in range(8):
        board[i] = input().rstrip("\n")
    query()
    turn = input().strip()

    wk = bk = wp = -1
    for i in range(8):
        for j in range(8):
            c = board[i][j]
            sq = i * 8 + j
            if c == "K":
                wk = sq
            elif c == "k":
                bk = sq
            elif c == "P":
                wp = sq

    side = 1 if turn == "B" else 0
    wkt = 0

    sid = id_map.get((wkt, wp, wk, bk, side), -1)
    if sid < 0 or ret[sid] < 0:
        print(0)
    else:
        plies = ret[sid]
        white_moves = (plies + 1) // 2
        print(white_moves)

# ////------------------ 스도쿠 ------------------////

class Node:
    def __init__(self, row=-1, col_header=None):
        self.left = self.right = self.up = self.down = self
        self.column = col_header
        self.size = 0
        self.row = row

def initialize_dancing_links(num_rows, num_cols):
    head = Node()
    head.left = head.right = head

    column_headers = []
    for c in range(num_cols):
        col_node = Node(row=-1, col_header=None)
        col_node.column = col_node
        col_node.size = 0
        col_node.up = col_node.down = col_node

        col_node.left = head.left
        col_node.right = head
        head.left.right = col_node
        head.left = col_node
        column_headers.append(col_node)

    row_header = [None] * num_rows
    return head, column_headers, row_header

def add_node(row, col, column_headers, row_header):
    col_node = column_headers[col]
    new_node = Node(row=row, col_header=col_node)

    new_node.down = col_node
    new_node.up = col_node.up
    col_node.up.down = new_node
    col_node.up = new_node
    col_node.size += 1

    if row_header[row] is None:
        row_header[row] = new_node
        new_node.left = new_node.right = new_node
    else:
        first = row_header[row]
        new_node.left = first
        new_node.right = first.right
        first.right.left = new_node
        first.right = new_node

def cover(col_node):
    col_node.right.left = col_node.left
    col_node.left.right = col_node.right
    row_node = col_node.down
    while row_node != col_node:
        right_node = row_node.right
        while right_node != row_node:
            right_node.down.up = right_node.up
            right_node.up.down = right_node.down
            right_node.column.size -= 1
            right_node = right_node.right
        row_node = row_node.down

def uncover(col_node):
    row_node = col_node.up
    while row_node != col_node:
        left_node = row_node.left
        while left_node != row_node:
            left_node.column.size += 1
            left_node.down.up = left_node
            left_node.up.down = left_node
            left_node = left_node.left
        row_node = row_node.up
    col_node.right.left = col_node
    col_node.left.right = col_node

def search(result, head):
    if head.right == head:
        return True

    c = head.right
    min_size = c.size
    ptr = c
    while ptr != head:
        if ptr.size < min_size:
            c = ptr
            min_size = ptr.size
        ptr = ptr.right
    cover(c)
    row_node = c.down
    while row_node != c:
        result.append(row_node.row)
        right_node = row_node.right
        while right_node != row_node:
            cover(right_node.column)
            right_node = right_node.right
        if search(result, head):
            return True

        result.pop()
        left_node = row_node.left
        while left_node != row_node:
            uncover(left_node.column)
            left_node = left_node.left
        row_node = row_node.down
    uncover(c)
    return False

def sudoku():
    query()
    board = [list(map(int, input().split())) for _ in range(9)]
    head, column_headers, row_header = initialize_dancing_links(729, 324)

    def constraint_indices(r, c, n):
        cell = r * 9 + c
        row_con = 81 + r * 9 + n
        col_con = 162 + c * 9 + n
        block = (r // 3) * 3 + (c // 3)
        block_con = 243 + block * 9 + n
        return [cell, row_con, col_con, block_con]

    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                n = board[r][c] - 1
                i = (r * 9 + c) * 9 + n
                for ci in constraint_indices(r, c, n):
                    add_node(i, ci, column_headers, row_header)

    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for n in range(9):
                    i = (r * 9 + c) * 9 + n
                    for ci in constraint_indices(r, c, n):
                        add_node(i, ci, column_headers, row_header)

    result = []
    if search(result, head):
        for i in result:
            r = i // 81
            c = (i // 9) % 9
            n = i % 9
            board[r][c] = n + 1

        for r in range(9):
            print(" ".join(map(str, board[r])))

# ////------------------ 나이트 투어 ------------------////

dx = [2, 1, -1, -2, -2, -1, 1, 2]
dy = [1, 2, 2, 1, -1, -2, -2, -1]
INF = 10**9

def chk(r, c, n):
    return 0 <= r < n and 0 <= c < n

def cdeg(r, c, n, vis):
    ret = 0
    for i in range(8):
        nr = r + dx[i]
        nc = c + dy[i]
        if chk(nr, nc, n) and not vis[nr][nc]:
            ret += 1
    return ret

def knights_tour():
    query()
    n, orr, oc = map(int, input().split())
    orr -= 1
    oc -= 1
    vis = [[False] * n for _ in range(n)]
    path = []
    mid = n + 1
    r, c = orr, oc

    for i in range(n * n):
        vis[r][c] = True
        path.append((r, c))
        if i == n * n - 1:
            break

        bdeg = INF
        bnr = -1
        bnc = -1
        bsc = -1

        for j in range(8):
            nr = r + dx[j]
            nc = c + dy[j]
            if not chk(nr, nc, n) or vis[nr][nc]:
                continue
            d = cdeg(nr, nc, n, vis)
            r1 = nr + 1
            c1 = nc + 1
            dr = 2 * r1 - mid
            dc = 2 * c1 - mid
            score = dr * dr + dc * dc
            if d < bdeg:
                bdeg = d
                bsc = score
                bnr = nr
                bnc = nc
            elif d == bdeg:
                if score > bsc:
                    bsc = score
                    bnr = nr
                    bnc = nc
                elif score == bsc:
                    if r1 < bnr + 1 or (r1 == bnr + 1 and c1 < bnc + 1):
                        bnr = nr
                        bnc = nc

        if bnr < 0:
            print(-1, -1)
            return

        r, c = bnr, bnc

    def row_label(rr):
        base = rr % 26
        quotient = rr // 26
        letter = chr(base + ord('a'))
        if quotient == 0:
            return letter
        else:
            return f"{letter}{quotient}'"

    for i, (rr, cc) in enumerate(path):
        print(row_label(rr) + str(cc + 1) + ('->' if i < len(path) - 1 else "."), end='')
    print()

reset = "\033[0m"
code = "\033[95m"
expl = "\033[93m"
ex = "\033[33m"
warn = "\033[91m"
intr = "\033[96m"

# ////------------------ 워들 ------------------////

def load_words(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_feedback(guess, solution):
    feedback = ['0'] * 5
    solution_chars = list(solution)
    used = [False] * 5

    for i in range(5):
        if guess[i] == solution[i]:
            feedback[i] = '2'
            used[i] = True

    for i in range(5):
        if feedback[i] == '0':
            for j in range(5):
                if not used[j] and guess[i] == solution_chars[j]:
                    feedback[i] = '1'
                    used[j] = True
                    break

    return ''.join(feedback)

def filter_candidates(candidates, guess, feedback):
    return [w for w in candidates if get_feedback(guess, w) == feedback]

def compute_entropy(guess, candidates):
    counts = {}
    total = len(candidates)
    for sol in candidates:
        pat = get_feedback(guess, sol)
        counts[pat] = counts.get(pat, 0) + 1

    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h

def choose_best_guess(all_guesses, candidates):
    if len(candidates) == 1:
        return candidates[0], 0.0

    best, besth = None, -1.0
    for g in all_guesses:
        h = compute_entropy(g, candidates)
        if h > besth:
            besth, best = h, g
    return best, besth

def wordle():
    try:
        all_words = load_words('words.txt')
    except Exception as e:
        print(f"{e}")
        return

    candidates = all_words.copy()
    guess = "salet"

    for turn in range(1, 7):
        if turn > 1:
            guess, _ = choose_best_guess(all_words, candidates)

        print(f"? {guess}")
        feedback = input().strip()
        if feedback == '22222':
            print(f"! {guess}")
            return

        candidates = filter_candidates(candidates, guess, feedback)
        if len(candidates) == 1:
            print(f"! {candidates[0]}")
            return
        if not candidates:
            print("쿼리 중 하나가 올바르지 않습니다.")
            return

    print("실패했습니다.")

# ////------------------ 틱택토 ------------------////

def evaluate_ttt(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '.':
            return 10 if board[i][0] == 'X' else -10
        if board[0][i] == board[1][i] == board[2][i] != '.':
            return 10 if board[0][i] == 'X' else -10
    if board[0][0] == board[1][1] == board[2][2] != '.':
        return 10 if board[0][0] == 'X' else -10
    if board[0][2] == board[1][1] == board[2][0] != '.':
        return 10 if board[0][2] == 'X' else -10
    return 0

def is_moves_left_ttt(board):
    for row in board:
        if '.' in row:
            return True
    return False

def minimax_ttt(board, depth, is_maximizing):
    score = evaluate_ttt(board)
    if score == 10:
        return score - depth
    if score == -10:
        return score + depth
    if not is_moves_left_ttt(board):
        return 0

    if is_maximizing:
        best = -sys.maxsize
        for i in range(3):
            for j in range(3):
                if board[i][j] == '.':
                    board[i][j] = 'X'
                    val = minimax_ttt(board, depth + 1, False)
                    board[i][j] = '.'
                    best = max(best, val)
        return best
    else:
        best = sys.maxsize
        for i in range(3):
            for j in range(3):
                if board[i][j] == '.':
                    board[i][j] = 'O'
                    val = minimax_ttt(board, depth + 1, True)
                    board[i][j] = '.'
                    best = min(best, val)
        return best

def find_best_move_ttt(board, player):
    best_move = (-1, -1)
    if player == 'X':
        best_val = -sys.maxsize
        for i in range(3):
            for j in range(3):
                if board[i][j] == '.':
                    board[i][j] = 'X'
                    move_val = minimax_ttt(board, 0, False)
                    board[i][j] = '.'
                    if move_val > best_val:
                        best_val = move_val
                        best_move = (i, j)
    else:
        best_val = sys.maxsize
        for i in range(3):
            for j in range(3):
                if board[i][j] == '.':
                    board[i][j] = 'O'
                    move_val = minimax_ttt(board, 0, True)
                    board[i][j] = '.'
                    if move_val < best_val:
                        best_val = move_val
                        best_move = (i, j)
    return best_move

def print_board_ttt(board):
    for row in board:
        for i in row:
            if i == 'X':
                print(f"\033[31mX\033[0m", end='')
            elif i == 'O':
                print(f"\033[34mO\033[0m", end='')
            else:
                print(".", end='')
        print()

def tic_tac_toe():
    logs = []
    while True:
        query()
        sym = sys.stdin.readline().strip().upper()
        if sym in ('X', 'O'):
            player_symbol = sym
            break
    computer_symbol = 'O' if player_symbol == 'X' else 'X'
    player_turn = (player_symbol == 'X')

    board = [['.' for _ in range(3)] for _ in range(3)]

    
    os.system(clear_cmd)
    for log in logs:
        query()
        print(log)
    print_board_ttt(board)

    while True:
        score = evaluate_ttt(board)
        if score == 10 or score == -10 or not is_moves_left_ttt(board):
            break

        if player_turn:
            
            while True:
                query()
                line = sys.stdin.readline().strip().split()
                if len(line) != 2:
                    continue
                try:
                    r = int(line[0])
                    c = int(line[1])
                except ValueError:
                    continue
                if not (0 <= r < 3 and 0 <= c < 3):
                    continue
                if board[r][c] != '.':
                    continue
                break
            board[r][c] = player_symbol
            logs.append(f"{r} {c}")
        else:
            
            r, c = find_best_move_ttt(board, computer_symbol)
            board[r][c] = computer_symbol
            logs.append(f"({r} {c})")

        
        os.system(clear_cmd)
        for log in logs:
            print(log)
        print_board_ttt(board)
        player_turn = not player_turn

    
    os.system(clear_cmd)
    for log in logs:
        print(log)
    print_board_ttt(board)

    final_score = evaluate_ttt(board)
    if final_score == 10:
        winner = 'X'
    elif final_score == -10:
        winner = 'O'
    else:
        winner = None

    if winner is None:
        print("무승부입니다.")
    else:
        if winner == player_symbol:
            print("플레이어가 승리했습니다!")
        else:
            print("컴퓨터가 승리했습니다!")

def explain():
    print("> 몇 번 메뉴를 알고싶으신가요? ", end="", flush=True)
    n = int(input().strip())
    if n == 1:
        print(f"기능: {ex}백이 폰 하나 가진 엔드 게임의 남은 수를 계산합니다.{reset}")
        print(f"{code}K(백의 킹), P(백의 폰), k(흑의 킹), .(빈칸){reset}으로 이루어진 8x8격자를 입력받습니다.")
        print(f"그 후 백과 흑 중 누구 차례인지 알 수 있도록 {code}W, B{reset} 중 하나를 입력합니다.")
        print(f"예:")
        print(f"{code}...k....\n...P....\n....K...\n........\n........\n........\n........\n........\nW{reset}")
        print(f"만일 {expl}출력 값이 0이라면, 그 게임은 무승부로 마무리 된다는 뜻{reset}입니다. 아닌 경우 {expl}출력값만큼의 수 안에 체크메이트가 된다는 뜻{reset}입니다.")
    elif n == 2:
        print(f"기능: {ex}8x8 스도쿠를 풉니다.{reset}")
        print(f"{code}0~9의 숫자{reset}로 이루어져 있는 띄어쓰기로 구분된 8x8 격자를 입력받습니다.")
        print(f"{code}0은 프로그램이 채워야할 칸{reset}을 의미합니다.")
        print(f"{expl}스도쿠의 정답 중 하나를 출력{reset}합니다.")
    elif n == 3:
        print(f"기능: {ex}N이 최소 6, 최대 666인 NxN 나이트투어를 해결합니다.{reset}")
        print(f"{code}첫 번째 줄에 N, 나이트의 위치의 행(숫자로 입력), 나이트의 현재 위치의 열{reset}을 띄어쓰기로 구분된 채로 입력받습니다.")
        print(f"{expl}나이트 투어의 정답을 출력{reset}합니다. 체스의 기보를 따르고, 행의 기보는 z다음은 a1', z1' 다음은 a2'식으로 진행됩니다.")
        print(f"{warn}이 메뉴는 휴리스틱하기 때문에 100%의 정확도를 보이지 않습니다. 하지만 97.7%의 정확도를 보입니다.{reset}")
    elif n == 4:
        print(f"{intr}이 메뉴는 인터랙티브입니다.{reset}")
        print(f"기능: {ex}뉴욕타임즈 단어에 기반한 워들을 풉니다.{reset}")
        print(f"{code}? 단어{reset} 로 컴퓨터가 먼저 단어를 제시합니다.")
        print(f"그 단어에 대해 피드백 해야합니다. {code}0, 1, 2 중 하나를 답해 모든 글자에 대하여 피드백 해야합니다.{reset}")
        print(f"{code}0은 단어에 그 글자가 들어있지 않다는 뜻{reset}이고, {code}1이면 단어에 그 글자가 있으나, 다른 위치에 있다는 뜻{reset}입니다. 마지막으로 {code}2는 그 글자가 그 위치에 있다는 것이 맞다는 뜻{reset}입니다.")
        print(f"이를 정답이 나올 때까지, 혹은 6번 반복하면 됩니다.")
        print(f"찾았다면 {expl}! 단어 를 출력{reset}합니다. 찾지 못했다면 {expl}실패했습니다. 를 출력{reset}합니다.")
        print(f"{warn}같은 글자가 여러번 나올 수 있음에 주의하세요.{reset}")
    elif n == 5:
        print(f"{intr}이 메뉴는 인터랙티브입니다.{reset}")
        print(f"기능: {ex}컴퓨터와 틱택토를 합니다.{reset}")
        print(f"{code}X, O{reset} 중 하나를 선택해 선공을 정합니다. X가 선공입니다.")
        print(f"플레이어가 공격일 경우 {code}행과 열{reset}을 띄어쓰기로 구분된 채로 입력받습니다.")
        print(f"{expl}게임을 진행하며 둔 쿼리를 출력{reset}합니다.")
        print(f"그 후 매 턴마다 {expl}현재 게임 판을 출력{reset}합니다.")
    elif n == 6:
        print(f"{warn}Lorem ipsum dolor sit amet, consectetur adipiscing elit. In sit amet odio velit. Donec eget congue nulla. Sed facilisis libero justo, feugiat faucibus leo placerat in. Sed luctus iaculis enim, quis luctus libero semper id. Sed imperdiet lacus eu nisl egestas facilisis. Nulla euismod ante enim, et dignissim justo pellentesque eu. Proin facilisis non leo sed maximus. Vestibulum odio ex, accumsan nec rhoncus non, sollicitudin non augue. Morbi sollicitudin augue in dolor consectetur luctus. Etiam at semper massa.{reset}")
    elif n == 7:
        print(f"{warn}O-05-47{reset}")

funcs = [KPvK, sudoku, knights_tour, wordle, tic_tac_toe, explain, None]

def main():
    try:
        if not id_map:
            
            print('die')
        os.system(clear_cmd)

        while True:
            print()
            print("메뉴")
            print("1. KPvK; 엔드게임 계산기")
            print("2. 스도쿠")
            print("3. 나이트 투어")
            print("4. 워들")
            print("5. 틱택토")
            print("6. 기능 설명")
            print("7. 종료")
            print("> 번호 입력: ", end="", flush=True)
            n = int(input().strip())

            if n == len(funcs):
                print("프로그램을 종료합니다.")
                break

            if 0 < n < len(funcs):
                if funcs[n-1] != explain:
                    os.system(clear_cmd)
                funcs[n-1]()
            else:
                print("올바른 번호를 입력하세요.")

            print("> 계속하려면 ENTER 혹은 Z를 누르세요 ", end="", flush=True)
            while True:
                key = sys.stdin.readline().strip().upper()
                if key == "" or key == "Z":
                    break
            os.system(clear_cmd)

    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
