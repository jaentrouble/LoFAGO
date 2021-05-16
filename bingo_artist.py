import numpy as np
import cv2

EMPTY = 0
FLIPPED = 1
BINGOED = 2
RECOMMAND = 3

SIZE = 640
MIDDLE = int(SIZE/2)
SPACE = int(SIZE/10)
SKULL_RAD = 43
SKULL_THIC = 5
RECOM_RAD = 30

BLACK=(0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)


class BingoArtist():
    def __init__(self) -> None:
        self.reset_board()


    def reset_board(self):
        board = np.ones((SIZE,SIZE,3),dtype=np.uint8)*255
        for i in range(6):
            board = cv2.line(board,
                            (0+SPACE*i,MIDDLE+SPACE*i),
                            (MIDDLE+SPACE*i,0+SPACE*i),
                            color=BLACK,
                            thickness=4)
        for i in range(6):
            board = cv2.line(board,
                            (MIDDLE-SPACE*i,0+SPACE*i),
                            (SIZE-SPACE*i,MIDDLE+SPACE*i),
                            color=BLACK,
                            thickness=4)
        self.board_img = board
        self.board = np.zeros((5,5),dtype=np.int)

    def draw_board(self, board, recommand=None, small=False):
        """draw_board
        update the board and get board image

        board : (5,5) numpy array
        recommand : (x,y)
        """
        self.reset_board()
        self.board[np.nonzero(board)] = FLIPPED
        # Check bingo
        x_bingo_idx = np.nonzero(np.logical_and.reduce(board, axis=1))[0]
        self.board[x_bingo_idx,:] = BINGOED
        y_bingo_idx = np.nonzero(np.logical_and.reduce(board,axis=0))[0]
        self.board[:,y_bingo_idx] = BINGOED
        if (board[0,0] and
            board[1,1] and
            board[2,2] and
            board[3,3] and
            board[4,4]):
            self.board[0,0] = self.board[1,1] = self.board[2,2] \
                            = self.board[3,3] = self.board[4,4] = BINGOED
        if (board[0,4] and
            board[1,3] and
            board[2,2] and
            board[3,1] and
            board[4,0]):
            self.board[0,4] = self.board[1,3] = self.board[2,2] \
                            = self.board[3,1] = self.board[4,0] = BINGOED
        
        flip_idx = self.board_idx_to_img_idx(*np.nonzero(self.board==FLIPPED))
        for x,y in zip(*flip_idx):
            self.board_img = cv2.circle(self.board_img,(x,y),
                                        SKULL_RAD,BLACK,SKULL_THIC)

        bingoed_idx = self.board_idx_to_img_idx(
                                *np.nonzero(self.board==BINGOED))
        for x,y in zip(*bingoed_idx):
            self.board_img = cv2.circle(self.board_img,(x,y),
                                        SKULL_RAD,RED,SKULL_THIC)
        
        if recommand is not None:
            rec_idx = self.board_idx_to_img_idx(*recommand)
            self.board_img = cv2.circle(self.board_img, rec_idx, 
                                        RECOM_RAD, BLUE, -1)

        if small:
            return cv2.resize(self.board_img, dsize=(0,0),fx=0.5, fy=0.5)
        else:
            return self.board_img.copy()

    def board_idx_to_img_idx(self, x_indices, y_indices):
        return (MIDDLE-SPACE*x_indices+SPACE*y_indices, 
                SPACE+SPACE*x_indices+SPACE*y_indices)
        

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    board = BingoArtist()
    test_board = np.array([
        [0,1,1,1,0],
        [0,0,0,1,0],
        [1,0,0,1,0],
        [0,0,0,1,0],
        [0,1,0,1,0]
    ])
    plt.imshow(board.draw_board(test_board,recommand=(1,1)))
    plt.show()