clear; clc;
f = fopen('Waypoints.txt', 'r');

line = fgetl(f);
i = 1;
x = [];
y = [];
while ~feof(f)
   line = strsplit(line, ',');
   x = [x, str2double(line{1})];
   y = [y, str2double(line{2})];
   line = fgetl(f);
end

fclose(f);