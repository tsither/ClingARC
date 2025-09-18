% Search for primitive shapes in input and output grids
% mapping to grids in (X,Y) are down then across

% #include "train_instances/1/grid_test.lp".


color_map(0,black). color_map(1,blue). color_map(2,red). color_map(3,green). color_map(4,yellow). color_map(5,gray). color_map(6,magenta). color_map(7,orange). color_map(8,cyan). color_map(9,brown).

filled_cell(InOut,(Source,Color)) :- cell(_,_,InOut,(Source,Int)), Int>0, color_map(Int,Color).

adjacent(InOut,Source,Target,Target_Color) :- filled_cell(InOut,(Source,Source_Color)),
    filled_cell(InOut,(Target,Target_Color)), 
    Source=(X1,Y1), Target=(X2,Y2),
    X1=X2, |Y1-Y2|=1.

adjacent(InOut,Source,Target,Target_Color) :- filled_cell(InOut,(Source,Source_Color)),
    filled_cell(InOut,(Target,Target_Color)), 
    Source=(X1,Y1), Target=(X2,Y2), 
    Y1=Y2, |X1-X2|=1.
    

%%% Primitives %%%

% 1. Points - isolated filled cells (not adjacent to any other filled cell)
point(InOut,Source,Color) :- filled_cell(InOut,(Source,Color)), not adjacent(InOut,Source,_,_).

% 2. Lines - sequences of 2 or more adjacent filled cells in a straight line (horizontal or vertical)

% Horizontal adjacency
h_adj(InOut,(X,Y),(X,Y+1),Color) :-
    filled_cell(InOut,((X,Y),Color)),
    filled_cell(InOut,((X,Y+1),Color)).

% Vertical adjacency
v_adj(InOut,(X,Y),(X+1,Y),Color) :-
    filled_cell(InOut,((X,Y),Color)),
    filled_cell(InOut,((X+1,Y),Color)).


% Extend a run
h_run(InOut,(X,Y),(X,Y2),Color,L) :- h_adj(InOut,(X,Y),(X,Y2),Color), L=2.

v_run(InOut,(X,Y),(X2,Y),Color,L) :- v_adj(InOut,(X,Y),(X2,Y),Color), L=2.

%% Recursively extend the run 
h_run(InOut,(X,Y),(X,Y3),Color,L+1) :- h_run(InOut,(X,Y),(X,Y2),Color,L), h_adj(InOut,(X,Y2),(X,Y3),Color).
v_run(InOut,(X,Y),(X3,Y),Color,L+1) :- v_run(InOut,(X,Y),(X2,Y),Color,L), v_adj(InOut,(X2,Y),(X3,Y),Color).

% Maximal horizontal line (no extension possible left/right)
horizontal_line(InOut,start(X,Y),end(X,Y2),color(Color),length(L)) :- h_run(InOut,(X,Y),(X,Y2),Color,L), L >= 2,
    not filled_cell(InOut,((X,Y-1),Color)),
    not filled_cell(InOut,((X,Y2+1),Color)).

% Maximal vertical line
vertical_line(InOut,start(X,Y),end(X2,Y),color(Color),length(L)) :- v_run(InOut,(X,Y),(X2,Y),Color,L), L >= 2,
    not filled_cell(InOut,((X-1,Y),Color)),
    not filled_cell(InOut,((X2+1,Y),Color)).

grid_size(InOut,Dx,Dy) :- grid(_,_,InOut,(Dx,Dy)).

% 3. Crosses - two intersecting lines (one horizontal, one vertical) of the same color
% cross(InOut,(X,Y),Color,L1,L2) :-


#show horizontal_line/5.
#show vertical_line/5.
#show point/3.
#show grid_size/3.



