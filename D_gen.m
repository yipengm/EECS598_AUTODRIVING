classes = {'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes', ...
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles', ...
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles', ...
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency', ...
    'Military', 'Commercial', 'Trains'};

%%
%files = dir('/home/yipengm/PycharmProjects/try598/deploy/trainval/0cec3d1f-544c-4146-8632-84b1f9fe89d3/0001_image.jpg');

files = dir('/home/yipengm/PycharmProjects/try598/deploy/test/*/*_image.jpg');

for idx = 1:numel(files)
  snapshot = [files(idx).folder, '/', files(idx).name];
  disp(snapshot)
  outfile = strrep(snapshot, '_image.jpg', '_depth.jpg');
  disp(outfile)
  img = imread(snapshot);

  xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
  xyz = reshape(xyz, [], 3)';

  proj = read_bin(strrep(snapshot, '_image.jpg', '_proj.bin'));
  proj = reshape(proj, [4, 3])';

  try
      bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
  catch
      disp('[*] no bbox found.')
      bbox = single([]);
  end
  bbox = reshape(bbox, 11, [])';

  uv = proj * [xyz; ones(1, size(xyz, 2))];
  depth = uv(3, :);
  uv = uv ./ uv(3, :);
  row = uv(2,:);
  col = uv(1,:);
  i1     = row>0;
  i2     = col>0;
  i3     = row<size(img,1);
  i4     = col<size(img,2);
  ix     = i1 & i2 & i3 & i4;

  depth = depth(ix);
  row = row(ix);
  col = col(ix);

  k = boundary(floor(col)',floor(row)',1);
  BW = poly2mask(col(k),row(k),size(img,1),size(img,2));
  %imshow(BW);
  %plot(col(k),row(k));
  %hold on
  thres = ceil(0.78897*size(img,1));
  
  
  [c_o,r_o] = meshgrid(1:1:size(img,2),1:1:size(img,1));
  d_o = griddata(col,row,depth,c_o,r_o);
  d_o_new = d_o.*BW+(1-BW)*80;
  d_o_new(isnan(d_o_new)) = 80;
  d_o_new(thres:size(img,1),:) = min(min(d_o_new));
  imwrite(d_o_new/max(max(d_o_new)),outfile);
  %imshow(d_o_new/max(max(d_o_new)));
end



%%
function [v, e] = get_bbox(p1, p2)
v = [p1(1), p1(1), p1(1), p1(1), p2(1), p2(1), p2(1), p2(1)
    p1(2), p1(2), p2(2), p2(2), p1(2), p1(2), p2(2), p2(2)
    p1(3), p2(3), p1(3), p2(3), p1(3), p2(3), p1(3), p2(3)];
e = [3, 4, 1, 1, 4, 4, 1, 2, 3, 4, 5, 5, 8, 8
    8, 7, 2, 3, 2, 3, 5, 6, 7, 8, 6, 7, 6, 7];
end

%%
function R = rot(n)
theta = norm(n, 2);
if theta
  n = n / theta;
  K = [0, -n(3), n(2); n(3), 0, -n(1); -n(2), n(1), 0];
  R = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
else
  R = eye(3);
end
end

%%
function data = read_bin(file_name)
id = fopen(file_name, 'r');
data = fread(id, inf, 'single');
fclose(id);
end
