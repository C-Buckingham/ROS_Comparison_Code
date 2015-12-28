black_T_Front = imread('71T01MBLK_large.jpg');
black_T_Back = imread('71T01MBLK_2_large.jpg');
cream_T_Front = imread('73G08MMUL_large.jpg');
cream_T_Back = imread('73G08MMUL_2_large.jpg');
black_And_Grey = imread('71S84LGRY_large.jpg');
grey_Diamond_Front = imread('71P67LGRY_large.jpg');
grey_Diamond_Back = imread('71P67LGRY_2_large.jpg');

for i = 1:3
    
    hist = imhist(cream_T_Front(:, :, i));
    hist2 = imhist(cream_T_Back(:, :, i));
    
    temp(i) = histogram_intersection(hist', hist2');
end

temp1 = 100-(sum(temp))/3*100;

black_T_Front = rgb2hsv(black_T_Front);
black_T_Back = rgb2hsv(black_T_Back);
cream_T_Front = rgb2hsv(cream_T_Front);
cream_T_Back = rgb2hsv(cream_T_Back);
black_And_Grey = rgb2hsv(black_And_Grey);
grey_Diamond_Front = rgb2hsv(grey_Diamond_Front);
grey_Diamond_Back = rgb2hsv(grey_Diamond_Back);

for i = 1:2
    
    hist = imhist(cream_T_Front(:, :, i));
    hist2 = imhist(cream_T_Back(:, :, i));
    
    temp(i) = histogram_intersection(hist', hist2');
end

temp2 = 100-(sum(temp))/2*100;

% t3 = round((t1+t2)/2)
(temp1+temp2)/2
if ((temp+temp2)/2 > 75)
    disp('T-Shirts are the same');
else
    disp('T-Shirts are different');    
end