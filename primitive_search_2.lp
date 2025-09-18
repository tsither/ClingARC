
% #include "instances/7/grid_0.lp".
% #include "train_instances/1/grid0.lp".
% #include "train_instances/16/grid_test.lp".

% #include "test_grid.lp".




%%% List of Primitives %%%
% 1. Points - 'isolated point' (not adjacent to any other filled cell)
%           - 'point' adjacent to a filled cell of a different color
% 2. Straight Lines - sequences of 2 or more adjacent filled cells in a straight line (horizontal or vertical)
% 3. Crosses - two intersecting lines (one horizontal, one vertical) of the same color
% 4. Intersection - two intersecting lines (one horizontal, one vertical) of different colors
% 5. Broken cross - a cross with a gap at the intersection (a point of a different color)
% 6. Diagonal lines - sequences of 2 or more adjacent filled cells in a diagonal line (45 degrees)
% 7. Square - four equal-length lines forming a square, inside not necessarily filled
% 8. Rectangle - four lines forming a rectangle (not square), inside not necessarily filled


#show grid_size/3.
#show isolated_point/3.
#show point/3.
#show square/5.
#show rectangle/6.
#show horizontal_line/5.
#show vertical_line/5.
#show diagonal_line/5.
#show cross/5.
#show broken_cross/4.
#show intersection/4.




%%% Primitives %%%

% Precursors and supplemental predicates

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
    
grid_size(InOut,Dx,Dy) :- grid(_,_,InOut,(Dx,Dy)).

% 1. Points - filled single cells (not adjacent to any other filled cell)
isolated_point(InOut,Source,Color) :- filled_cell(InOut,(Source,Color)), not adjacent(InOut,Source,_,_), 
    not d_adj_dr(InOut,Source,_,_),
    not d_adj_ur(InOut,Source,_,_),
    not d_adj_dr(InOut,_,Source,_),
    not d_adj_ur(InOut,_,Source,_).

% point(InOut,Source,Color1) :- filled_cell(InOut,(Source,Color1)), adjacent(InOut,Source,Target,Color2), Color1!=Color2, Source!=Target.
point(InOut,Source,Color1) :- filled_cell(InOut,(Source,Color1)), adjacent(InOut,Source,Target,Color2), Color1!=Color2, Source!=Target, not adjacent(InOut,Source,_,Color1).

% 2. Straight Lines - sequences of 2 or more adjacent filled cells in a straight line (horizontal or vertical)
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
horizontal_line_(InOut,start(X,Y),end(X,Y2),color(Color),length(L)) :- h_run(InOut,(X,Y),(X,Y2),Color,L), L >= 2,
    not filled_cell(InOut,((X,Y-1),Color)),
    not filled_cell(InOut,((X,Y2+1),Color)).

% Maximal vertical line
vertical_line_(InOut,start(X,Y),end(X2,Y),color(Color),length(L)) :- v_run(InOut,(X,Y),(X2,Y),Color,L), L >= 2,
    not filled_cell(InOut,((X-1,Y),Color)),
    not filled_cell(InOut,((X2+1,Y),Color)).

% 3. Crosses - two intersecting lines (one horizontal, one vertical) of the same color

intersection_of_cross(InOut,(X,Y),Color) :- 
    horizontal_line_(InOut,start(X,Y1),end(X,Y2),color(Color),length(L1)),
    vertical_line_(InOut,start(X1,Y),end(X2,Y),color(Color),length(L2)),
    Y1 < Y, Y < Y2,  % Y coordinate is within horizontal line range
    X1 < X, X < X2.  % X coordinate is within vertical line range

% #show intersection_of_cross/3.

cross_(InOut,center(X,Y),vertical(top(X_v1,Y_v),bottom(X_v2,Y_v)),horizontal(left(X_h,Y_h1),right(X_h,Y_h2)), color(Color)) :- 
    vertical_line_(InOut,start(X_v1,Y_v),end(X_v2,Y_v),color(Color),length(L1)),
    horizontal_line_(InOut,start(X_h,Y_h1),end(X_h,Y_h2),color(Color),length(L2)),
    intersection_of_cross(InOut,(X,Y),Color),
    Y_h1 < Y, Y < Y_h2,  % Y coordinate is within horizontal line range
    X_v1 < X, X < X_v2.  % X coordinate is within vertical line range


% 4. Intersection - two intersecting lines (one horizontal, one vertical) of different colors

h_line_break(InOut,center((X,Y),Color),horizontal(length(Total_L),H_Color)) :- 
    % point(InOut,(X,Y),Color), % center break must be a 'point'
    filled_cell(InOut,((X,Y),Color)),
    horizontal_line_(InOut,start(X,H1_start),end(X,H1_end),color(H_Color),length(H_L1)),
    horizontal_line_(InOut,start(X,H2_start),end(X,H2_end),color(H_Color),length(H_L2)),
    H1_end + 1= Y, Y + 1= H2_start,
    Total_L = H_L1 + H_L2.


v_line_break(InOut,center((X,Y),Color),vertical(length(Total_L),V_Color)) :- 
    % point(InOut,(X,Y),Color), % center break must be a 'point'
    filled_cell(InOut,((X,Y),Color)),
    vertical_line_(InOut,start(V1_start,Y),end(V1_end,Y),color(V_Color),length(V_L1)),
    vertical_line_(InOut,start(V2_start,Y),end(V2_end,Y),color(V_Color),length(V_L2)),
    V1_end + 1= X, X + 1= V2_start,
    Total_L = V_L1 + V_L2.


intersection(InOut,center((X,Y),Center_color),vertical(length(L1),V_Color),horizontal(length(L2),H_Color)) :- 
    h_line_break(InOut,center((X,Y),Center_color),horizontal(length(L2),H_Color)),
    v_line_break(InOut,center((X,Y),Center_color),vertical(length(L1),V_Color)),
    H_Color != V_Color.

% 5. Broken cross - a cross with a gap at the intersection (a point of a different color)
broken_cross(InOut,center((X,Y),Center_color),vertical(length(L2),Color),horizontal(length(L1),Color)) :- 
    h_line_break(InOut,center((X,Y),Center_color),horizontal(length(L1),Color)),
    v_line_break(InOut,center((X,Y),Center_color),vertical(length(L2),Color)).


% 6. Diagnal lines - sequences of 2 or more adjacent filled cells in a diagonal line (45 degrees)
% Diagonal adjacency (down-right)
d_adj_dr(InOut,(X,Y),(X+1,Y+1),Color) :-
    filled_cell(InOut,((X,Y),Color)),
    filled_cell(InOut,((X+1,Y+1),Color)).

% Diagonal adjacency (up-right)
d_adj_ur(InOut,(X,Y),(X-1,Y+1),Color) :-
    filled_cell(InOut,((X,Y),Color)),
    filled_cell(InOut,((X-1,Y+1),Color)).

% Diagonal run (up-right)
d_run_ur(InOut,(X,Y),(X2,Y2),Color,L) :- d_adj_ur(InOut,(X,Y),(X2,Y2),Color), L = 2.
d_run_ur(InOut,(X,Y),(X3,Y3),Color,L+1) :- d_run_ur(InOut,(X,Y),(X2,Y2),Color,L), d_adj_ur(InOut,(X2,Y2),(X3,Y3),Color), L>=2.

% Diagonal run (down-right)
d_run_dr(InOut,(X,Y),(X2,Y2),Color,L) :- d_adj_dr(InOut,(X,Y),(X2,Y2),Color), L = 2.
d_run_dr(InOut,(X,Y),(X3,Y3),Color,L+1) :- d_run_dr(InOut,(X,Y),(X2,Y2),Color,L), d_adj_dr(InOut,(X2,Y2),(X3,Y3),Color), L>=2.


% Diagonal line (up-right)
diagonal_line_(InOut,start(X,Y),end(X2,Y2),color(Color),length(L)) :- d_run_ur(InOut,(X,Y),(X2,Y2),Color,L), 
    not filled_cell(InOut,((X2-1,Y2+1),Color)),
    not filled_cell(InOut,((X+1,Y-1),Color)).

% Diagonal line (down-right)
diagonal_line_(InOut,start(X,Y),end(X2,Y2),color(Color),length(L)) :- d_run_dr(InOut,(X,Y),(X2,Y2),Color,L), 
    not filled_cell(InOut,((X2+1,Y2+1),Color)), % no down right (2,2) and no (3,3)
    not filled_cell(InOut,((X-1,Y-1),Color)).  % no up left (1,1) and no (0,0)

% 7. Square 
square(InOut,top_left_corner(X,Y),bottom_right_corner(X+Size-1,Y+Size-1),size(Size),color(Color)) :- 
    horizontal_line_(InOut,start(X,Y),end(X,Y+Size-1),color(Color),length(Size)),                    % Top edge
    horizontal_line_(InOut,start(X+Size-1,Y),end(X+Size-1,Y+Size-1),color(Color),length(Size)),      % Bottom edge
    vertical_line_(InOut,start(X,Y),end(X+Size-1,Y),color(Color),length(Size)),                      % Left edge
    vertical_line_(InOut,start(X,Y+Size-1),end(X+Size-1,Y+Size-1),color(Color),length(Size)),        % Right edge
    Size >= 2.

% 8. Rectangle
rectangle(InOut,top_left_corner(X,Y),bottom_right_corner(X+Height-1,Y+Width-1),height(Height),width(Width),color(Color)) :- 
    horizontal_line_(InOut,start(X,Y),end(X,Y+Width-1),color(Color),length(Width)),                  % Top edge  
    horizontal_line_(InOut,start(X+Height-1,Y),end(X+Height-1,Y+Width-1),color(Color),length(Width)), % Bottom edge
    vertical_line_(InOut,start(X,Y),end(X+Height-1,Y),color(Color),length(Height)),                  % Left edge
    vertical_line_(InOut,start(X,Y+Width-1),end(X+Height-1,Y+Width-1),color(Color),length(Height)),  % Right edge
    Width >= 2, Height >= 2, Width != Height.



%%%%%%% REDUNDANCY CHECK %%%%%%%%
%%% Only show shapes that aren't already in other shapes %%%
% Reduce redundancy in output, don't flood LLM with too many redundant shapes
% 1. lower primitives in squares and rectangles should not be shown
% 2. lines that are part of crosses should not be shown

in_square_or_rectangle(InOut,(X,Y),Color) :- 
    square(InOut,top_left_corner(X1,Y1),bottom_right_corner(X2,Y2),size(Size),color(Color)),
    X1<=X, X<=X2, Y1<=Y, Y<=Y2, 
    filled_cell(InOut,((X,Y),Color)).


in_square_or_rectangle(InOut,(X,Y),Color) :- 
    rectangle(InOut,top_left_corner(X1,Y1),bottom_right_corner(X2,Y2),height(Height),width(Width),color(Color)),
    X1<=X, X<=X2, Y1<=Y, Y<=Y2, 
    filled_cell(InOut,((X,Y),Color)).

in_cross(InOut,(X,Y),Color) :- 
    cross_(InOut,center(_,_),vertical(top(X_v1,Y_v),bottom(X_v2,Y_v)),horizontal(left(X_h,Y_h1),right(X_h,Y_h2)), color(Color)),
    X_v1<=X, X<=X_v2, Y_h1<=Y, Y<=Y_h2, 
    filled_cell(InOut,((X,Y),Color)).

% in_broken_cross(InOut,(X,Y),Color) :- 
%     broken_cross(InOut,center(_,_),vertical(length(L1),V_Color),horizontal(length(L2),H_Color)),
%     (Color = V_Color; Color = H_Color),
%     filled_cell(InOut,((X,Y),Color)).

% in_intersection(InOut,(X,Y),Color) :- 
%     intersection(InOut,center((X,Y),Center_color),vertical(length(L1),V_Color),horizontal(length(L2),H_Color)),
%     (Color = V_Color; Color = H_Color),
%     filled_cell(InOut,((X,Y),Color)).



%%%% Final Primitives to be shown %%%%

% Horizontal lines - only show if not part of square/rectangle
horizontal_line(InOut,start(X,Y),end(X,Y2),color(Color),length(L)) :- 
    horizontal_line_(InOut,start(X,Y),end(X,Y2),color(Color),length(L)), 
    not in_square_or_rectangle(InOut,(X,Y),Color),
    not in_square_or_rectangle(InOut,(X,Y2),Color),
    not in_cross(InOut,(X,Y),Color),
    not in_cross(InOut,(X,Y2),Color).


% Vertical lines - only show if not part of square/rectangle
vertical_line(InOut,start(X,Y),end(X2,Y),color(Color),length(L)) :- 
    vertical_line_(InOut,start(X,Y),end(X2,Y),color(Color),length(L)),
    not in_square_or_rectangle(InOut,(X,Y),Color),
    not in_square_or_rectangle(InOut,(X2,Y),Color),
    not in_cross(InOut,(X,Y),Color),
    not in_cross(InOut,(X2,Y),Color).

% Diagonal lines - only show if not part of square/rectangle or cross
diagonal_line(InOut,start(X,Y),end(X2,Y2),color(Color),length(L)) :- 
    diagonal_line_(InOut,start(X,Y),end(X2,Y2),color(Color),length(L)),
    not in_square_or_rectangle(InOut,(X,Y),Color),
    not in_square_or_rectangle(InOut,(X2,Y2),Color),
    not in_cross(InOut,(X,Y),Color),
    not in_cross(InOut,(X2,Y2),Color).

% Crosses - only show if not part of square/rectangle
cross(InOut,center(X,Y),vertical(top(X_v1,Y_v),bottom(X_v2,Y_v)),horizontal(left(X_h,Y_h1),right(X_h,Y_h2)), color(Color)) :- 
    cross_(InOut,center(X,Y),vertical(top(X_v1,Y_v),bottom(X_v2,Y_v)),horizontal(left(X_h,Y_h1),right(X_h,Y_h2)), color(Color)),
    % % not redunant(InOut,(X,Y),Color),
    % not redunant(InOut,(X_v1,Y_v),Color),
    % not redunant(InOut,(X_v2,Y_v),Color),
    % not redunant(InOut,(X_h,Y_h1),Color),
    % not redunant(InOut,(X_h,Y_h2),Color).
    not in_square_or_rectangle(InOut,(X,Y),Color),
    not in_square_or_rectangle(InOut,(X_v1,Y_v),Color),
    not in_square_or_rectangle(InOut,(X_v2,Y_v),Color),
    not in_square_or_rectangle(InOut,(X_h,Y_h1),Color),
    not in_square_or_rectangle(InOut,(X_h,Y_h2),Color).



