import utils
import os
import math
import json
import jsbeautifier
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.channels import read_custom_montage
from scipy.interpolate import Rbf
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

def get_matched(tpl_names, tpl_dict):
	return [name for name in tpl_names if tpl_dict[name]["matched"]==True]

def get_empty_template(tpl_names, tpl_dict):
	return [name for name in tpl_names if tpl_dict[name]["matched"]==False]

def get_unassigned_input(in_names, in_dict):
	return [name for name in in_names if in_dict[name]["assigned"]==False]

def read_montage(loc_file):
	tpl_montage = read_custom_montage("./template_chanlocs.loc")
	in_montage = read_custom_montage(loc_file)
	tpl_names = tpl_montage.ch_names
	in_names = in_montage.ch_names
	tpl_dict = {}
	in_dict = {}
	
	# convert all channel names to uppercase and store their information
	for i, name in enumerate(tpl_names):
		up_name = str.upper(name)
		tpl_montage.rename_channels({name: up_name})
		tpl_dict[up_name] = {
			"index" : i,
			"coord_3d" : tpl_montage.get_positions()['ch_pos'][up_name],
			"matched" : False
		}
	for i, name in enumerate(in_names):
		up_name = str.upper(name)
		in_montage.rename_channels({name: up_name})
		in_dict[up_name] = {
			"index" : i,
			"coord_3d" : in_montage.get_positions()['ch_pos'][up_name],
			"assigned" : False
		}
	return tpl_montage, in_montage, tpl_dict, in_dict

def match_name(stage1_info):
	# read the location file
	loc_file = stage1_info["fileNames"]["inputData"]
	tpl_montage, in_montage, tpl_dict, in_dict = read_montage(loc_file)
	tpl_names = tpl_montage.ch_names
	in_names = in_montage.ch_names
	old_idx = [[None]]*30 # store the indices of the in_channels in the order of tpl_channels
	is_orig_data = [False]*30
	
	alias_dict = {
		'T3': 'T7',
		'T4': 'T8',
		'T5': 'P7',
		'T6': 'P8'
	}
	for i, name in enumerate(tpl_names):
		if name in alias_dict and alias_dict[name] in in_dict:
			tpl_montage.rename_channels({name: alias_dict[name]})
			tpl_dict[alias_dict[name]] = tpl_dict.pop(name)
			name = alias_dict[name]
			
		if name in in_dict:
			old_idx[i] = [in_dict[name]["index"]]
			is_orig_data[i] = True
			tpl_dict[name]["matched"] = True
			in_dict[name]["assigned"] = True
	
	# update the names
	tpl_names = tpl_montage.ch_names
	
	stage1_info.update({
		"unassignedInput" : get_unassigned_input(in_names, in_dict),
		"emptyTemplate" : get_empty_template(tpl_names, tpl_dict),
	 	"mappingResult" : [
		{
			"index" : old_idx,
			"isOriginalData" : is_orig_data
		}
		]
	})
	channel_info = {
		"templateNames" : tpl_names,
		"inputNames" : in_names,
		"templateDict" : tpl_dict,
		"inputDict" : in_dict
	}
	return stage1_info, channel_info, tpl_montage, in_montage

def align_coords(channel_info, tpl_montage, in_montage):
	tpl_names = channel_info["templateNames"]
	in_names = channel_info["inputNames"]
	tpl_dict = channel_info["templateDict"]
	in_dict = channel_info["inputDict"]
	matched_names = get_matched(tpl_names, tpl_dict)
	
	# 2D alignment (for visualization purposes)
	fig = [tpl_montage.plot(), in_montage.plot()]
	ax = [fig[0].axes[0], fig[1].axes[0]]
	
	# extract the displayed 2D coordinates
	all_tpl = ax[0].collections[0].get_offsets().data
	all_in= ax[1].collections[0].get_offsets().data
	matched_tpl = np.array([all_tpl[tpl_dict[name]["index"]] for name in matched_names])
	matched_in = np.array([all_in[in_dict[name]["index"]] for name in matched_names])
	plt.close('all')
	
	# apply TPS to transform in_channels to align with tpl_channels positions
	rbf_x = Rbf(matched_in[:,0], matched_in[:,1], matched_tpl[:,0], function='thin_plate')
	rbf_y = Rbf(matched_in[:,0], matched_in[:,1], matched_tpl[:,1], function='thin_plate')
	
	# apply the transformation to all in_channels
	transformed_in_x = rbf_x(all_in[:,0], all_in[:,1])
	transformed_in_y = rbf_y(all_in[:,0], all_in[:,1])
	transformed_in = np.vstack((transformed_in_x, transformed_in_y)).T
	
	for i, name in enumerate(tpl_names):
		tpl_dict[name]["coord_2d"] = all_tpl[i]
	for i, name in enumerate(in_names):
		in_dict[name]["coord_2d"] = transformed_in[i].tolist()
	
	
	# 3D alignment
	all_tpl = np.array([tpl_dict[name]["coord_3d"].tolist() for name in tpl_names])
	all_in = np.array([in_dict[name]["coord_3d"].tolist() for name in in_names])
	matched_tpl = np.array([all_tpl[tpl_dict[name]["index"]] for name in matched_names])
	matched_in = np.array([all_in[in_dict[name]["index"]] for name in matched_names])
	
	rbf_x = Rbf(matched_in[:,0], matched_in[:,1], matched_in[:,2], matched_tpl[:,0], function='thin_plate')
	rbf_y = Rbf(matched_in[:,0], matched_in[:,1], matched_in[:,2], matched_tpl[:,1], function='thin_plate')
	rbf_z = Rbf(matched_in[:,0], matched_in[:,1], matched_in[:,2], matched_tpl[:,2], function='thin_plate')
	
	transformed_in_x = rbf_x(all_in[:,0], all_in[:,1], all_in[:,2])
	transformed_in_y = rbf_y(all_in[:,0], all_in[:,1], all_in[:,2])
	transformed_in_z = rbf_z(all_in[:,0], all_in[:,1], all_in[:,2])
	transformed_in = np.vstack((transformed_in_x, transformed_in_y, transformed_in_z)).T
	
	for i, name in enumerate(in_names):
		in_dict[name]["coord_3d"] = transformed_in[i].tolist()
	
	channel_info.update({
		"templateDict" : tpl_dict,
		"inputDict" : in_dict
	})
	return channel_info

def save_figure(channel_info, tpl_montage, filename1, filename2):
	tpl_names = channel_info["templateNames"]
	in_names = channel_info["inputNames"]
	tpl_dict = channel_info["templateDict"]
	in_dict = channel_info["inputDict"]
	
	tpl_x = [tpl_dict[name]["coord_2d"][0] for name in tpl_names]
	tpl_y = [tpl_dict[name]["coord_2d"][1] for name in tpl_names]
	in_x = [in_dict[name]["coord_2d"][0] for name in in_names]
	in_y = [in_dict[name]["coord_2d"][1] for name in in_names]
	tpl_coords = np.vstack((tpl_x, tpl_y)).T
	in_coords = np.vstack((in_x, in_y)).T
	
	# extract template's head figure
	tpl_fig = tpl_montage.plot()
	tpl_ax = tpl_fig.axes[0]
	lines = tpl_ax.lines
	head_lines = []
	for line in lines:
		x, y = line.get_data()
		head_lines.append((x,y))
	
	# -------------------------plot input montage------------------------------
	fig = plt.figure(figsize=(6.4,6.4), dpi=100)
	ax = fig.add_subplot(111)
	fig.tight_layout()
	ax.set_aspect('equal')
	ax.axis('off')
	
	# plot template's head
	for x, y in head_lines:
		ax.plot(x, y, color='black', linewidth=1.0)
	# plot in_channels on it
	ax.scatter(in_coords[:,0], in_coords[:,1], s=35, color='black')
	for i, name in enumerate(in_names):
		ax.text(in_coords[i,0]+0.004, in_coords[i,1], name, color='black', fontsize=10.0, va='center')
	# save input_montage
	fig.savefig(filename1)
	
	# ---------------------------add indications-------------------------------
	# plot unmatched input channels in red
	indices = [in_dict[name]["index"] for name in in_names if in_dict[name]["assigned"]==False]
	if indices != []:
		ax.scatter(in_coords[indices,0], in_coords[indices,1], s=35, color='red')
		for i in indices:
			ax.text(in_coords[i,0]+0.004, in_coords[i,1], in_names[i], color='red', fontsize=10.0, va='center')
	# save mapped_montage
	fig.savefig(filename2)
	
	# -------------------------------------------------------------------------
	# store the tpl and in_channels' display positions (in px)
	tpl_coords = ax.transData.transform(tpl_coords)
	in_coords = ax.transData.transform(in_coords)
	plt.close('all')
	
	for i, name in enumerate(tpl_names):
		left = tpl_coords[i,0]/6.4
		bottom = tpl_coords[i,1]/6.4
		tpl_dict[name]["css_position"] = [round(left, 2), round(bottom, 2)]
	for i, name in enumerate(in_names):
		left = in_coords[i,0]/6.4
		bottom = in_coords[i,1]/6.4
		in_dict[name]["css_position"] = [round(left, 2), round(bottom, 2)]
	
	channel_info.update({
		"templateDict" : tpl_dict,
		"inputDict" : in_dict
	})
	return channel_info

def find_neighbors(channel_info, empty_tpl_names, old_idx):
	in_names = channel_info["inputNames"]
	tpl_dict = channel_info["templateDict"]
	in_dict = channel_info["inputDict"]
	
	all_in = [np.array(in_dict[name]["coord_3d"]) for name in in_names]
	empty_tpl = [np.array(tpl_dict[name]["coord_3d"]) for name in empty_tpl_names]
	
	# use KNN to choose k nearest channels
	k = 4 if len(in_names)>4 else len(in_names)
	knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
	knn.fit(all_in)
	for i, name in enumerate(empty_tpl_names):
		distances, indices = knn.kneighbors(empty_tpl[i].reshape(1,-1))
		idx = tpl_dict[name]["index"]
		old_idx[idx] = indices[0].tolist()
	
	return old_idx

def optimal_mapping(channel_info):
	tpl_names = channel_info["templateNames"]
	in_names = channel_info["inputNames"]
	tpl_dict = channel_info["templateDict"]
	in_dict = channel_info["inputDict"]
	unass_in_names = get_unassigned_input(in_names, in_dict)
	# reset all tpl.matched to False
	for name in tpl_dict:
		tpl_dict[name]["matched"] = False
	
	all_tpl = np.array([tpl_dict[name]["coord_3d"] for name in tpl_names])
	unass_in = np.array([in_dict[name]["coord_3d"] for name in unass_in_names])
	
	# initialize the cost matrix for the Hungarian algorithm
	if len(unass_in_names) < 30:
		cost_matrix = np.full((30, 30), 1e6) # add dummy channels to ensure num_col >= num_row
	else:
		cost_matrix = np.zeros((30, len(unass_in_names)))
	# fill the cost matrix with Euclidean distances between tpl and unassigned in_channels
	for i in range(30):
		for j in range(len(unass_in_names)):
			cost_matrix[i][j] = np.linalg.norm((all_tpl[i]-unass_in[j])*1000)
	
	# apply the Hungarian algorithm to optimally assign one in_channel to each tpl_channel
	# by minimizing the total distances between their positions.
	row_idx, col_idx = linear_sum_assignment(cost_matrix)
	
	# store the mapping result
	old_idx = [[None]]*30
	is_orig_data = [False]*30
	for i, j in zip(row_idx, col_idx):
		if j < len(unass_in_names): # filter out dummy channels
			tpl_name = tpl_names[i]
			in_name = unass_in_names[j]

			old_idx[i] = [in_			dict[in_name]["index"]]
			is_orig_data[i] = True
			tpl_dict[tpl_name]["matched"] = True
			in_dict[in_name]["assigned"] = True
	
	# fill the remaining empty tpl_channels
	empty_tpl_names = get_empty_template(tpl_names, tpl_dict)
	if empty_tpl_names != []:
		old_idx = find_neighbors(channel_info, empty_tpl_names, old_idx)
	
	result = {
		"index" : old_idx,
		"isOriginalData" : is_orig_data
	}
	channel_info["inputDict"] = in_dict
	return result, channel_info

def mapping_result(stage1_info, channel_info, filename):
	unassigned_num = len(stage1_info["unassignedInput"])
	batch = math.ceil(unassigned_num/30) + 1
	
	# map the remaining in_channels
	results = stage1_info["mappingResult"]
	for i in range(1, batch):
		# optimally select 30 in_channels to map to the tpl_channels based on proximity
		result, channel_info = optimal_mapping(channel_info)
		results += [result]
	'''
	for i in range(batch):
		results[i]["name"] = {}
		for j, indices in enumerate(results[i]["index"]):
			names = [channel_info["inputNames"][idx] for idx in indices] if indices!=[None] else ["zero"]
			results[i]["name"][channel_info["templateNames"][j]] = names
	'''
	data = {
		#"templateNames" : channel_info["templateNames"],
		#"inputNames" : channel_info["inputNames"],
		"channelNum" : len(channel_info["inputNames"]),
		"batch" : batch,
		"mappingResult" : results
	}
	options = jsbeautifier.default_options()
	options.indent_size = 4
	json_data = jsbeautifier.beautify(json.dumps(data), options)
	with open(filename, 'w') as jsonfile:
		jsonfile.write(json_data)
	
	stage1_info.update({
		"batch" : batch,
		"mappingResult" : results
	})
	return stage1_info, channel_info

