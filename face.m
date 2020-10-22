clear; clc; clf; close all

dn1 = './CroppedYale/';
dn2 = './yalefaces_uncropped/yalefaces/';
dn_fig = './figs/';

dir1 = dir(fullfile(dn1,'yaleB*'));

%% read + reshape (CROPPED)

% 1. For face space of ONE PERSON, set 'i_subdir' and uncomment 
%    'for ss = i_subdir:i_subdir' for the loop condition
% 2. For face space of ALL PEOPLE, uncomment 'for ss = 1:length(dir1)'


A = []; % data matrix
i_subdir = 1;

for ss = 1:length(dir1)
%for ss = i_subdir:i_subdir
    disp(ss)
    sdn = dir1(ss).name;
    sdir = dir(fullfile([dn1,sdn],'*.pgm'));
    
    % compile all images in each subdir
    for ff = 1:length(sdir)
        fn = sdir(ff).name;
        full_fn = [dn1,sdn,'/',fn];
        
        A_ff = double(imread(full_fn, 'pgm')); % image matrix
        A_ff_c = reshape(A_ff, length(A_ff(:)), 1); % reshape into column
        A = [A, A_ff_c]; % append to A
        clear A_ff_c
    end
    
    % display average image
    figure(1)
    imshow(uint8(reshape(mean(A, 2), size(A_ff))))
    title(gca, 'average face of selected images')
    
    clear A_ff
end

disp('A compiled');

%% svd (CROPPED)
% method 1
[U1, S1, V1] = svd(A, 'econ'); 

% method 2
%C = A*A'; % corr. matrix
%[U2, S2] = eigs(C, 20, 'lm');
%clear C

%% plot cols of U - eigenfaces (CROPPED)
fig2 = figure(2);
set(fig2,'Position',[100 100 1000 1000])

nRows = 3; nCols = 3;

for pp = 1:(nRows*nCols-1)
    subplot(nRows,nCols,pp)
    imshow(uint8(2.5e4.*reshape(U1(:,pp),192,168)))
end

subplot(nRows,nCols,nRows*nCols)
semilogy(1:nRows*nCols, diag(S1(1:nRows*nCols,1:nRows*nCols)));
xlim([1,nRows*nCols])
xlabel('singular value index')
ylabel('singular value')

%% find rank (CROPPED)
Ev = diag(S1);
tol = 0.95*sum(Ev.^2);
sum_sigma = 0;
ee = 0; % rank index

while sum_sigma <= tol
    ee = ee + 1;
    sum_sigma = sum_sigma + Ev(ee)^2;
end

disp(ee) % rank

fig3 = figure(3);
imshow(uint8(2.5e4.*reshape(U1(:,ee), 192, 168)))
title(gca, [num2str(ee),'th eigenface'])

%% plot eigs of AA' (CROPPED)
fig4 = figure(4);
set(fig2,'Position',[100 100 1000 1000])

nRows = 3; nCols = 3;

for pp = 1:(nRows*nCols-1)
    subplot(nRows,nCols,pp)
    imshow(reshape(V2(:,pp).*100,192,168))
end

subplot(nRows,nCols,nRows*nCols)
semilogy(1:nRows*nCols, diag(S2(1:nRows*nCols,1:nRows*nCols)));
xlim([1,nRows*nCols])
xlabel('eigenvalue index')
ylabel('eigenvalue')
%save_compact_pdf(fig2, [dir_fig,'efaces','20'])

%% read + reshape (UNCROPPED)

% 2. For face space of ALL PEOPLE, uncomment 'for ss = 1:length(dir1)'

A_uc= []; % uncropped data matrix

% 1. For face space of ONE SUBJECT, set 'i_subject' and uncomment here
i_subject = 1;
%dir2 = dir(fullfile(dn2,['subject', num2str(i_subject,'%02d'), '*']));

% 2. For face space of ONE CONDITION, set 'condn' and uncomment here
condn = 'normal';
%dir2 = dir(fullfile(dn2,['subject*.', condn]));

% 2. For face space of ALL SUBJECTS, uncomment here
dir2 = dir(fullfile(dn2,'subject*'));

for ss = 1:length(dir2)
    fn = dir2(ss).name;
    disp(fn)
    full_fn = [dn2,fn];
    
    A_uc_ff = double(imread(full_fn, 'gif')); % image matrix
    A_uc_ff_c = reshape(A_uc_ff, length(A_uc_ff(:)), 1); % reshape into column
    A_uc = [A_uc, A_uc_ff_c]; % append to A
    clear A_uc_ff_c
    
    % display average image
    figure(1)
    imshow(uint8(reshape(mean(A_uc, 2), 243,320)))
    title(gca, 'average face of selected images')
    
    clear A_uc_ff
end

%% svd (UNCROPPED)
% method 1
[U1, S1, V1] = svd(A_uc, 'econ'); 

%% plot cols of U - eigenfaces (UNCROPPED)
fig2 = figure(2);
set(fig2,'Position',[100 100 1000 1000])

nRows = 3; nCols = 3;

for pp = 1:(nRows*nCols-1)
    subplot(nRows,nCols,pp)
    imshow(uint8(2.5e4.*reshape(U1(:,pp),243,320)))
end

subplot(nRows,nCols,nRows*nCols)
semilogy(1:nRows*nCols, diag(S1(1:nRows*nCols,1:nRows*nCols)));
xlim([1,nRows*nCols])
xlabel('singular value index')
ylabel('singular value')

%% find rank (UNCROPPED)
Ev = diag(S1);
tol = 0.98*sum(Ev.^2);
sum_sigma = 0;
ee = 0; % rank index

while sum_sigma <= tol
    ee = ee + 1;
    sum_sigma = sum_sigma + Ev(ee)^2;
end

disp(ee) % rank

fig3 = figure(3);
imshow(uint8(2.5e4.*reshape(U1(:,ee), 243, 320)))
title(gca, [num2str(ee),'th eigenface'])

%% proj. different sets of images
% V*a