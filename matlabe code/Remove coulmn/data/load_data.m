%%  IN the NAME OF ALLAH
function [x] =load_data(n)
%% loud data
%----------------------------------orl-------------------------------------
% % % class number = 40
% % % observition each class = 10
%--------------------------------- FEI-------------------------------------
% % % class number = 200
% % % observition each class = 14
% --------------------------------Yeal-------------------------------------
% % % class number = 15
% % % observition each class = 11
%--------------------------------coil_20-----------------------------------
% % % class number = 20
% % % observition each class = 72
%-------------------------------- digits--------------------------------------
% % % x_train = x{1,1} label_train = {1,2}
% % % x_test  = x{2,1] label_test  = {2,2}
%--------------------------------AR-------------------------------------------
% % % class number = 101
% % % observition each class = 26
% ----------------------------------------------------------------------------
switch n
    case 1
        % orginal image Yale
        load('Yale_database')
        x = Yeal;
    case 2
        % manually crop Yale
        load('crop_Yale_database')
        x = Yale;
    case 3
        % orl data
        load('orl_database1')
        x = double(orl_database1);
    case 4
        % coil_20 data
        load('coil_20')
        x = coil_20;
    case 5
        % orginal FEI data
        load('FEI')
        x = FEI;
    case 6
        % digits data
        load('digit')
        x = {tr_imag,tr_label;te_imag,te_label};
    case 7
        % AR data set
         load('AR')
         x = double(AR);
    case 8
        % manully crop AR
         load('AR_crop_manully')
         x = double(AR_crop_manully);
end