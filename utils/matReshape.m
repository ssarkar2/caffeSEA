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
    
    data_1 = logical(eval(inmats{data_ind}));
    data_2 = logical(eval(inmats{label_ind}));

    whos
    clear(inmats{data_ind})
    clear(inmats{label_ind})

    whos

    [a,b,c] = size(data_1)
    out_mat = reshape(data_1, [a,1,b,c]);
    whos


   
    %[a,b,c] = size(data_1)
    %out_mat = false(a, 1, b, c);  %using false instead of zeros, since the data is logical/boolean. may save space and make it faster
    %out_mat(:,1,:,:) = data_1;

    save_file = struct(inmats{data_ind}, out_mat, inmats{label_ind}, data_2);

    %save(strrep(input_matrix, '.mat', '_swap.mat'), '-struct', 'save_file', '-v7.3');
    save(input_matrix, '-struct', 'save_file', '-v7.3');
    quit
end
