function matReshape(inpParams)
    t = strsplit(inpParams, ',');
    input_matrix = t{1}; outDir = t{2}; chunksz = str2num(t{3});
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

    chunkCount = 0;
    num_total_samples = size(data_1,1)
    for batchno=1:num_total_samples/chunksz
        chunkCount = chunkCount + 1
        fprintf('batch no. %d\n', batchno);
        last_read=(batchno-1)*chunksz;

        if last_read+chunksz < size(data_2,1)
            endidx = last_read+chunksz;
        else
            endidx = size(data_2,1);
        end
        
        x = data_1(last_read+1:endidx, :, :);
        [a,b,c] = size(x);
        x = permute(x, [3 2 1]);
        batchdata = reshape(x,[c b 1 a]);
        
        y = data_2(last_read+1:endidx, :);
        [a,b] = size(y);
        y = permute(y, [2 1]);  %transpose
        batchlabs = reshape(y, [b a]);
        
        % store to hdf5
        startloc=struct('data',[1,1,1,1], 'label', [1,1]);
        filename = getFileName(input_matrix, outDir, batchno);
        curr_dat_sz=store2hdf5(filename, batchdata, batchlabs, startloc, chunksz); 
        h5disp(filename);
    end
    save([outDir 'chunkCount.mat'],'chunkCount')
    quit;
end

function nm = getFileName(input_matrix, outDir, batchno)
  t = strsplit(input_matrix, '/');
  t = strsplit(t{end}, '.');
  dataType = t{1};  %train test or valid 
  nm = [outDir dataType num2str(batchno) '.hdf5'];
end

function [curr_dat_sz, curr_lab_sz] = store2hdf5(filename, data, labels, startloc, chunksz)  
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
  % *label* is D*N matrix of labels (D labels per sample) 
  % chunksz (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations 

  % verify that format is right
  dat_dims=size(data);
  lab_dims=size(labels);
  num_samples=dat_dims(end);

  assert(lab_dims(end)==num_samples, 'Number of samples should be matched between data and labels');

  %fprintf('Creating dataset with %d samples\n', num_samples);
  if ~exist('chunksz', 'var')
      chunksz=1000;
  end
  if exist(filename, 'file')
      fprintf('Warning: replacing existing file %s \n', filename);
      delete(filename);
  end
  filename
  h5create(filename, '/data', [dat_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [dat_dims(1:end-1) chunksz]); % width, height, channels, number
  h5create(filename, '/label', [lab_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [lab_dims(1:end-1) chunksz]); % width, height, channels, number
  if ~exist('startloc','var')
      startloc.data=[ones(1,length(dat_dims)-1), 1];
      startloc.label=[ones(1,length(lab_dims)-1), 1];
  end


  if ~isempty(data)
    h5write(filename, '/data', single(data), startloc.data, size(data));
    h5write(filename, '/label', single(labels), startloc.label, size(labels));  
  end

  if nargout
    info=h5info(filename);
    curr_dat_sz=info.Datasets(1).Dataspace.Size;
    curr_lab_sz=info.Datasets(2).Dataspace.Size;
  end
end
