function matReshape(input_matrix)
    
    load(input_matrix);
    inmats = who();
    
    data_ind = 0;
    label_ind = 0;
    for i = 1:length(inmats)
        if (length(strfind(inmats{i}, 'data'))) == 1
            if (length(strfind(inmats{i}, 'xdata'))) == 1
                data_ind = i;
            else
                label_ind = i;
            end
        end
    end
    
    data_1 = eval(inmats{data_ind});
    data_2 = eval(inmats{label_ind});
   
    [a,b,c] = size(data_1);
    out_mat = zeros(a, 1, b, c);
    out_mat(:,1,:,:) = data_1;
    save_file = struct(inmats{data_ind}, out_mat, inmats{label_ind}, data_2);

    save(input_matrix, '-struct', 'save_file');
    quit
end
