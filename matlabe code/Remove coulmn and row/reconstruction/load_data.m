%%  IN the NAME OF ALLAH
function [x] =load_data(n)
%% loud data

switch n
    case 1
        % orginal image Yale 
         load('F:\thesis\database\img-database\data\Yale_database')
         x = Yeal;
    case 2
        % manually crop Yale 
         load('F:\thesis\database\img-database\data\crop_Yale_database') 
         x = Yale;
    case 3
         % orl data         
         load('F:\thesis\database\img-database\data\orl_database1') 
         x = double(orl_database1);
    case 4
        % coil_20 data
         load('F:\thesis\database\img-database\data\coil_20') 
         x = coil_20;
    case 5
        % orginal FEI data
        load('F:\thesis\database\img-database\data\FEI') 
        x = FEI;
    case 6
        % digits data
        load('F:\thesis\database\img-database\data\digit') 
        x = {tr_imag,tr_label;te_imag,te_label};
end