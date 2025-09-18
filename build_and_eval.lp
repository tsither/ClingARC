% Build and evaluate output grid from primitives %

% #include "instances/7/grid_test.lp".

% grid_size(output,3,3).
% % vertical_line(output,start(1,0),end(2,0),color(red),length(2)).
% vertical_line(output,start(1,0),end(2,0),color(red)).

% grid_size(output,9,9).
% vertical_line(output,start(0,4),end(5,4),color(cyan),length(6)).
% horizontal_line(output,start(1,2),end(1,8),color(cyan),length(7)).
% vertical_line(output,start(2,1),end(8,1),color(orange),length(7)).
% horizontal_line(output,start(6,0),end(6,3),color(orange),length(4)).
% point(output,(1,1),red).
% point(output,(6,4),red).
% point(output,(1,0),cyan).
% point(output,(0,1),orange).
% vertical_line(output,start(7,4),end(8,4),color(cyan),length(2)).
% horizontal_line(output,start(6,5),end(6,8),color(orange),length(4)).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build output grid from primitives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define every cell in grid for safety
cell_(output,(X,Y)) :- grid_size(output,Dx,Dy), X=0..Dx-1, Y=0..Dy-1.

color_map(0,black). color_map(1,blue). color_map(2,red). color_map(3,green). color_map(4,yellow). color_map(5,gray). color_map(6,magenta). color_map(7,orange). color_map(8,cyan). color_map(9,brown).

%%% Translate primitives to cell predictions %%%
% Translate point to cell facts (pre-cursor to cell_prediction)
cell_prediction_(output,Cell,Color) :- point(output,Cell,color(Color)).
cell_prediction_(output,Cell,Color) :- point(output,Cell,Color).

cell_prediction_(output,Cell,Color) :- isolated_point(output,Cell,color(Color)).
cell_prediction_(output,Cell,Color) :- isolated_point(output,Cell,Color).

% Translate horizontal_line to cell facts (pre-cursor to cell_prediction)
cell_prediction_(output,(X,Y1..Y2),Color) :- horizontal_line(output,start(X,Y1),end(X,Y2),color(Color),length(L)).

% Translate vertical_line to cell facts (pre-cursor to cell_prediction)
cell_prediction_(output,(X1..X2,Y),Color) :- vertical_line(output,start(X1,Y),end(X2,Y),color(Color),length(L)).

% Translate square to cell facts (pre-cursor to cell_prediction)
cell_prediction_(output,(X1..X2,Y1),Color) :- square(output,top_left_corner(X1,Y1),bottom_right_corner(X2,Y2),size(Size),color(Color)).
cell_prediction_(output,(X1..X2,Y2),Color) :- square(output,top_left_corner(X1,Y1),bottom_right_corner(X2,Y2),size(Size),color(Color)).

cell_prediction_(output,(X1,Y1..Y2),Color) :- square(output,top_left_corner(X1,Y1),bottom_right_corner(X2,Y2),size(Size),color(Color)).
cell_prediction_(output,(X2,Y1..Y2),Color) :- square(output,top_left_corner(X1,Y1),bottom_right_corner(X2,Y2),size(Size),color(Color)).

% Identify cells with no information (no relevant primitives)
no_info(output,(X,Y)) :- not cell_prediction_(output,(X,Y),_), cell_(output,(X,Y)).

%%% Create final cell facts %%%
% create black cells
cell_prediction(output,(X,Y),0) :- no_info(output,(X,Y)).

% Create final cells from predictions
cell_prediction(output,(X,Y),Int) :- cell_prediction_(output,(X,Y),Color), color_map(Int,Color).

% mismatch(Output,(X,Y),C1,C2) :- cell_prediction(Output,(X,Y),C1), cell_prediction(Output,(X,Y),C2), C1 != C2.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate accuracy of output grid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Case 1: if there is a celll that is the wrong color
incorrect_cell((X,Y),(predicted(Color),correct(Color1))) :- cell(0,test,output,((X,Y),C1)), cell_prediction(output,(X,Y),C), C != C1,
    color_map(C,Color), color_map(C1,Color1).

% Case 2: if there is a cell in the expected output that has no prediction
incorrect_cell((X,Y),missed(Color)) :- cell(0,test,output,((X,Y),C)), not cell_prediction(output,(X,Y),_), color_map(C,Color).

% Case 3: duplicate prediction
incorrect_cell((X,Y),duplicate(Color1,Color2)) :- 
    cell_prediction_(output,(X,Y),Color1), 
    cell_prediction_(output,(X,Y),Color2), 
    Color1 != Color2.

% Case 4: if there is a prediction for a cell that shouldn't exist in the expected output
incorrect_cell((X,Y),extra(Color)) :- cell_prediction(output,(X,Y),C), not cell(0,test,output,((X,Y),_)), color_map(C,Color).


all_correct :- correct_grid.

incorrect_grid :- incorrect_cell(_,_).
correct_grid :- not incorrect_grid.

#show incorrect_grid/0.
#show correct_grid/0.
#show incorrect_cell/2.
% #show cell_prediction_/3.
