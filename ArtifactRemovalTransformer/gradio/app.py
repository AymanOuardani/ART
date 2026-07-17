import utils
import app_utils
import os
import uuid
import tempfile
import gradio as gr

gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = gradio_temp_dir


title = """
<div>
	<div style="display: flex; justify-content: center; text-align: center; font-size: 2rem;">
		<b>Artifact Removal Transformer 🤗 Gradio Demo</b>
	</div>
	<br>
	<div style="display: flex; justify-content: center; text-align: center;">
		<p>
		<b>ART: Artifact Removal Transformer for Reconstructing Noise-Free Multichannel Electroencephalographic Signals</b>
		<br>
		Chun-Hsiang Chuang, Kong-Yi Chang, Chih-Sheng Huang, Anne-Mei Bessas
		</p>
	</div>
	<br>
	<div style="display: flex; justify-content: center; column-gap: 4px;">
		<a href='https://arxiv.org/abs/2409.07326' target='_blank'">
			<img src='https://img.shields.io/badge/arXiv-paper-red'>
		</a>
		<a href='https://github.com/CNElab-Plus/ArtifactRemovalTransformer' target='_blank'>
			<img src='https://img.shields.io/badge/GitHub-code-blue'>
		</a>
		<a href='https://sites.google.com/view/chchuang' target='_blank'>
			<img src='https://img.shields.io/badge/CNElab-contact-9b27b1'>
		</a>
	</div>
</div>
"""

guide = """

This 🤗 Gradio Demo is designed to assist you with two main tasks:
1. **Channel Mapping**: Align your EEG channels with our template channels to ensure compatibility with our models.
2. **EEG Artifact Removal**: Use our models—**ART**, **IC-U-Net**, **IC-U-Net++**, and **IC-U-Net-Attn**—to denoise your EEG data.

## File Requirements and Preparation
- **Channel locations**: If you don't have the channel location file, we recommend you to download the standard montage <a href="">here</a>. If the channels in those files don't match yours, you can use **EEGLAB** to adjust them to your required montage.
- **Raw data**: Your data format must be a two-dimensional array (channels, timepoints).<br>
❗️❗️❗️Your data must include some channels that correspond to our template channels, which include: **Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCz, FC4, FT8, T7, C3, Cz, C4, T8, TP7, CP3, CPz, CP4, TP8, P7, P3, Pz, P4, P8, O1, Oz, O2**. At least some of them need to be present for successful mapping.<br>
❗️❗️❗️Please remove any reference, ECG, EOG, EMG, or other non-EEG channels before uploading your files.

## Step1. Channel Mapping
The following steps will guide you through the process of mapping your EEG channels to our template channels.

### Step1-1: Initial Matching and Scaling
After clicking on `Map` button, we will first match your channels to our template channels by their names. Using the matched channels as reference points, we will apply Thin Plate Spline (TPS) transformation to scale your montage to align with our template's dimensions. The template montage and your scaled montage will be displayed side by side for comparison. Channels that do not have a match in our template will be **highlighted in red**.
- If your data includes all the 30 template channels, you will be directed to **Mapping Result**.
- If your data doesn't include all the 30 template channels and you have some channels that do not match the template, you will be directed to **Step2**.
- If all your channels are included in our template but you have fewer than 30 channels, you will be directed to **Step3**.

### Step1-2: Forwarding Unmatched Channels
In this step, you will handle the channels that didn't have a direct match with our template, by manually assigning them to the template channels that are still empty, ensuring the most efficient use of your data.<br>
Your unmatched channels, previously highlighted in red, will be shown on your montage with a radio button displayed above each. You can choose to forward the data from these unmatched channels to the empty template channels. The interface will display each empty template channel in sequence, allowing you to select which of your unmatched channels to forward.
- If all empty template channels are filled by your selections, you will be directed to **Mapping Result**.
- If there are still empty template channels remaining, you will be directed to **Step3**.

### Step1-3: Filling Remaining Template Channels
To run the models successfully, we need to ensure that all 30 template channels are filled. In this step, you are required to select one of the methods provided below to fill the remaining empty template channels:
- **Mean** method: Each empty template channel is filled with the average value of data from the nearest input channels. By default, the 4 closest input channels (determined after aligning your montage to the template's scale using TPS) are selected for this averaging process. On the interface, you will see checkboxes displayed above each of your channel. The 4 nearest channels are pre-selected by default for each empty template channel, but you can modify these selections as needed. If you uncheck all the checkboxes for a particular template channel, it will be filled with zeros.
- **Zero** method: All empty template channels are filled with zeros.<br>
Choose the method that best suits your needs, considering that the model's performance may vary depending on the method used.<br>
Once all template channels are filled, you will be directed to **Mapping Result**.

### Mapping Result
After completing the previous steps, your channels will be aligned with the template channels required by our models.
- In case there are still some channels that haven't been mapped, we will automatically batch and optimally assign them to the template. This ensures that even channels not initially mapped will still be included in the final result.
- Once the mapping process is completed, a JSON file containing the mapping result will be generated. This file is necessary only if you plan to run the models using the source code; otherwise, you can ignore it.

## Step2. Data Denoising
After uploading your EEG data and clicking on `Run` button, we will process your data based on the mapping result.<br>
- If necessary, your data will be divided into batches and run the models on each batch sequentially, ensuring that all channels are properly processed.
"""

icunet = """
## IC-U-Net
### Abstract
Electroencephalography (EEG) signals are often contaminated with artifacts. It is imperative to develop a practical and reliable artifact removal method to prevent the misinterpretation of neural signals and the underperformance of brain–computer interfaces. Based on the U-Net architecture, we developed a new artifact removal model, IC-U-Net, for removing pervasive EEG artifacts and reconstructing brain signals. IC-U-Net was trained using mixtures of brain and non-brain components decomposed by independent component analysis. It uses an ensemble of loss functions to model complex signal fluctuations in EEG recordings. The effectiveness of the proposed method in recovering brain activities and removing various artifacts (e.g., eye blinks/movements, muscle activities, and line/channel noise) was demonstrated in a simulation study and four real-world EEG experiments. IC-U-Net can reconstruct a multi-channel EEG signal and is applicable to most artifact types, offering a promising end-to-end solution for automatically removing artifacts from EEG recordings. It also meets the increasing need to image natural brain dynamics in a mobile setting.

C.-H. Chuang, K.-Y. Chang, C.-S. Huang, and T.-P. Jung, "IC-U-Net: A U-Net-based denoising autoencoder using mixtures of independent components for automatic EEG artifact removal," NeuroImage, vol. 263, p. 119586, 2022/11/01/ 2022
"""

icunetpp = """
## IC-U-Net++
### Abstract
Electroencephalographic (EEG) data is considered contaminated with various types of artifacts. Deep learning has been successfully applied to developing EEG artifact removal techniques to increase the signal-to-noise ratio (SNR) and enhance brain-computer interface performance. Recently, our research team has proposed an end-to-end UNet-based EEG artifact removal technique, IC-U-Net, which can reconstruct signals against various artifacts. However, this model suffers from being prone to overfitting with a limited training dataset size and demanding a high computational cost. To address these issues, this study attempted to leverage the architecture of UNet++ to improve the practicability of IC-U-Net by introducing dense skip connections in the encoder-decoder architecture. Results showed that this proposed model obtained superior SNR to the original model with half the number of parameters. Also, this proposed model achieved comparable convergency using a quarter of the training data size.

K. Y. Chang, Y. C. Huang, and C. H. Chuang, "Enhancing EEG Artifact Removal Efficiency by Introducing Dense Skip Connections to IC-U-Net," in 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 24-27 July 2023 2023, pp. 1-4
"""

eegart="""
## ART
### Abstract
Artifact removal in electroencephalography (EEG) is a longstanding challenge that significantly impacts neuroscientific analysis and brain-computer interface (BCI) performance. Tackling this problem demands advanced algorithms, extensive noisy-clean training data, and thorough evaluation strategies. This study presents the Artifact Removal Transformer (ART), an innovative EEG denoising model employing transformer architecture to adeptly capture the transient millisecond-scale dynamics characteristic of EEG signals. Our approach offers a holistic, end-to-end denoising solution for diverse artifact types in multichannel EEG data. We enhanced the generation of noisy-clean EEG data pairs using an independent component analysis, thus fortifying the training scenarios critical for effective supervised learning. We performed comprehensive validations using a wide range of open datasets from various BCI applications, employing metrics like mean squared error and signal-to-noise ratio, as well as sophisticated techniques such as source localization and EEG component classification. Our evaluations confirm that ART surpasses other deep-learning-based artifact removal methods, setting a new benchmark in EEG signal processing. This advancement not only boosts the accuracy and reliability of artifact removal but also promises to catalyze further innovations in the field, facilitating the study of brain dynamics in naturalistic environments.
"""

js = """
() => {
	const styleSheet = document.styleSheets[0];
	styleSheet.insertRule(`
		.channel-box {
			position: absolute;
			z-index: 2;
			width: 2.5%;
			height: 2.5%;
			transform: translate(-50%, 50%);
		}
	`, styleSheet.cssRules.length);
	styleSheet.insertRule(`
		.channel-input {
			display: block !important;
			width: 100% !important;
			height: 100% !important;
		}
	`, styleSheet.cssRules.length);
}
"""

init_js = """
(stage1_info, channel_info) => {
	stage1_info = JSON.parse(JSON.stringify(stage1_info));
	channel_info = JSON.parse(JSON.stringify(channel_info));
	
	let selector, attribute;
	if(stage1_info.state == "step2-selecting"){
		selector = "#radio-group > div:nth-of-type(2)";
		attribute = "value";
	}else if(stage1_info.state == "step3-2-selecting"){
		selector = "#chkbox-group > div:nth-of-type(2)";
		attribute = "name";
	}else return;
	
	const div = document.querySelector(selector);
	
	// add figure of the input montage
	div.style.cssText = `
		position: relative;
		width: 100%;
		aspect-ratio: 1;
		background-image: url("file=${stage1_info.fileNames.originalMontage}");
		background-position: left bottom;
		background-size: 100%;
	`;
	
	// move the radios/checkboxes
	let name, left, bottom;
	const elements = div.querySelectorAll(":scope > label");
	Array.from(elements).forEach( el => {
		name = el.querySelector(":scope > input").getAttribute(attribute);
		left = channel_info.inputDict[name].css_position[0];
		bottom = channel_info.inputDict[name].css_position[1];
		
		el.className = "channel-box";
		el.style.cssText = `left: ${left}%; bottom: ${bottom}%;`;
		el.querySelector(":scope > input").classList.add("channel-input");
		el.querySelector(":scope > span").innerText = "";
	});
	
	// add indication for the first empty tpl_channel
	name = stage1_info.emptyTemplate[0];
	left = channel_info.templateDict[name].css_position[0];
	bottom = channel_info.templateDict[name].css_position[1];
	const dotRule = `
		${selector}::before {
			content: "";
			position: absolute;
			left: ${left}%;
			bottom: ${bottom}%;
			width: 2%;
			height: 2%;
			border-radius: 50%;
			background-color: red;
		}
	`;
	const textRule = `
		${selector}::after {
			content: "${name}";
			position: absolute;
			z-index: 1;
			left: ${left+2.7}%;
			bottom: ${bottom}%;
			font-size: 1em;
			font-weight: 900;
			color: red;
		}
	`;
	// check if indicator already exist
	const styleSheet = document.styleSheets[0];
	for(let i=0; i<styleSheet.cssRules.length; i++){
		let tmp = styleSheet.cssRules[i].selectorText;
		if(tmp==selector+"::before" || tmp==selector+"::after"){
			styleSheet.deleteRule(i);
			i--;
		}
	}
	styleSheet.insertRule(dotRule, styleSheet.cssRules.length);
	styleSheet.insertRule(textRule, styleSheet.cssRules.length);
}
"""

update_js = """
(stage1_info, channel_info) => {
	stage1_info = JSON.parse(JSON.stringify(stage1_info));
	channel_info = JSON.parse(JSON.stringify(channel_info));
	
	let selector;
	let cnt, name, left, bottom;
	if(stage1_info.state == "step2-selecting"){
		selector = "#radio-group > div:nth-of-type(2)";
		cnt = stage1_info.step2.count;
		
		// update the radios
		const elements = document.querySelectorAll(selector+" > label");
		Array.from(elements).forEach( el => {
			name = el.querySelector(":scope > input").value;
			left = channel_info.inputDict[name].css_position[0];
			bottom = channel_info.inputDict[name].css_position[1];
			el.style.cssText = `left: ${left}%; bottom: ${bottom}%;`;
		});
	}else if(stage1_info.state == "step3-2-selecting"){
		selector = "#chkbox-group > div:nth-of-type(2)";
		cnt = stage1_info.step3.count;
	}else return;
	
	// update the indication
	name = stage1_info.emptyTemplate[cnt-1];
	left = channel_info.templateDict[name].css_position[0];
	bottom = channel_info.templateDict[name].css_position[1];
	const dotRule = `
		${selector}::before {
			content: "";
			position: absolute;
			left: ${left}%;
			bottom: ${bottom}%;
			width: 2%;
			height: 2%;
			border-radius: 50%;
			background-color: red;
		}
	`;
	const textRule = `
		${selector}::after {
			content: "${name}";
			position: absolute;
			z-index: 1;
			left: ${left+2.7}%;
			bottom: ${bottom}%;
			font-size: 1em;
			font-weight: 900;
			color: red;
		}
	`;
	
	// update the rules
	const styleSheet = document.styleSheets[0];
	for(let i=0; i<styleSheet.cssRules.length; i++){
		let tmp = styleSheet.cssRules[i].selectorText;
		if(tmp==selector+"::before" || tmp==selector+"::after"){
			styleSheet.deleteRule(i);
			i--;
		}
	}
	styleSheet.insertRule(dotRule, styleSheet.cssRules.length);
	styleSheet.insertRule(textRule, styleSheet.cssRules.length);
}
"""

with gr.Blocks(js=js, delete_cache=(3600, 3600)) as demo:
	session_dir = gr.State("")
	stage1_json = gr.JSON({}, visible=False)
	stage2_json = gr.JSON({}, visible=False)
	channel_json = gr.JSON({}, visible=False)

	gr.HTML(title)
	with gr.Row():

		with gr.Column(variant="panel"):
			gr.Markdown("## Step1. Channel Mapping")
			# ---------------------input---------------------
			in_loc_file = gr.File(label="Channel locations (.loc, .locs, .xyz, .sfp, .txt)",
							file_types=[".loc", "locs", ".xyz", ".sfp", ".txt"])
			map_btn = gr.Button("Map")
			# ---------------------output--------------------
			desc_md = gr.Markdown(visible=False)
			out_result_file = gr.File(visible=False)
			# --------------------mapping--------------------
			# step1-1
			with gr.Row():
				tpl_img = gr.Image("./template_montage.png", label="Template montage", visible=False)
				mapped_img = gr.Image(label="Matching result", visible=False)
			# step1-2
			radio_group = gr.Radio(elem_id="radio-group", visible=False)
			# step1-3
			with gr.Row():
				in_fillmode = gr.Dropdown(choices=["mean", "zero"],
											value="mean",
											label="Filling method",
											visible=False,
											scale=2)
				fillmode_btn = gr.Button("OK", visible=False, scale=1)
			chkbox_group = gr.CheckboxGroup(elem_id="chkbox-group", visible=False)
			
			with gr.Row():
				clear_btn = gr.Button("Clear", visible=False)
				step2_btn = gr.Button("Next", visible=False)
				step3_btn = gr.Button("Next", visible=False)
				next_btn = gr.Button("Next step", visible=False)
			# -----------------------------------------------
		
		with gr.Column(variant="panel"):
			gr.Markdown("## Step2. Data Denoising")
			# ---------------------input---------------------
			with gr.Row():
				in_data_file = gr.File(label="Raw data (.csv)", file_types=[".csv"])
				with gr.Column():
					in_samplerate = gr.Textbox(label="Sampling rate (Hz)")
					in_modelname = gr.Dropdown(choices=[
										("ART", "ART"),
										("IC-U-Net", "ICUNet"),
										("IC-U-Net++", "ICUNet++"),
										("IC-U-Net-Attn", "ICUNet_attn")],
										value="ART",
										label="Model")
					run_btn = gr.Button("Run", interactive=False)
					cancel_btn = gr.Button("Cancel", visible=False)
			# ---------------------output--------------------
			batch_md = gr.Markdown(visible=False)
			out_data_file = gr.File(label="Denoised data", visible=False)
			# -----------------------------------------------
			
	with gr.Row():
		with gr.Tab("User Guide"):
			gr.Markdown(guide)
		with gr.Tab("ART"):
			gr.Markdown(eegart)
		with gr.Tab("IC-U-Net"):
			gr.Markdown(icunet)
		with gr.Tab("IC-U-Net++"):
			gr.Markdown(icunetpp)
		with gr.Tab("IC-U-Net-Attn"):
			gr.Markdown()
	
	def create_dir(req: gr.Request):
		os.mkdir(gradio_temp_dir+'/'+req.session_hash+'/')
		return gradio_temp_dir+'/'+req.session_hash+'/'
	demo.load(create_dir, inputs=[], outputs=session_dir)
	
	# +========================================================================================+
	# |                                Stage1: channel mapping                                 |
	# +========================================================================================+
	def reset_all(rootpath, stage1_info, stage2_info, in_loc):
		if in_loc == None:
			gr.Warning("Please upload a file.")
			stage1_info["errorFlag"] = True
			return {stage1_json : stage1_info}
		
		# delete the previous folder of Stage1, 2
		if "filePath" in stage1_info:
			utils.dataDelete(stage1_info["filePath"])
		if "filePath" in stage2_info and stage2_info.get("state")!="stopped":
			utils.dataDelete(stage2_info["filePath"])
		# establish a new folder
		stage1_dir = uuid.uuid4().hex + '_stage1/'
		os.mkdir(rootpath + stage1_dir)
		
		inputname = os.path.basename(str(in_loc))
		outputname = inputname[:-4] + '_mapping_result.json'
		
		stage1_info = {
			"filePath" : rootpath + stage1_dir,
			"fileNames" : {
				"inputData" : in_loc,
				"originalMontage" : rootpath + stage1_dir + 'input_montage.png',
				"mappedMontage" : rootpath + stage1_dir + 'mapped_montage.png',
				"outputData" : rootpath + stage1_dir + outputname
			},
			"state" : "step1-initializing",
			"errorFlag" : False,
			"step2" : {
				"count" : None,
				"totalNum" : None
			},
			"step3" : {
				"count" : None,
				"totalNum" : None
			},
			"unassignedInput" : None,
			"emptyTemplate" : None,
			"batch" : None,
			"mappingResult" : [
			{
				"index" : None,
				"isOriginalData" : None
				#"channelUsageNum" : None
			}
			]
		}
		stage2_info = {}
		channel_info = {}
		return {stage1_json : stage1_info,
				stage2_json : stage2_info,
				channel_json : channel_info,
				# --------------------Stage1-------------------------
				map_btn : gr.Button(interactive=False),
				desc_md : gr.Markdown("", visible=True),
				out_result_file : gr.File(value=None, visible=False),
				tpl_img : gr.Image(visible=False),
				mapped_img : gr.Image(value=None, visible=False),
				radio_group : gr.Radio(choices=[], value=[], label="", visible=False),
				in_fillmode : gr.Dropdown(value="mean", visible=False),
				fillmode_btn : gr.Button(visible=False),
				chkbox_group : gr.CheckboxGroup(choices=[], value=[], label='', visible=False),
				clear_btn : gr.Button(visible=False),
				step2_btn : gr.Button(visible=False),
				step3_btn : gr.Button(visible=False),
				next_btn : gr.Button(visible=False),
				# --------------------Stage2-------------------------
				in_data_file : gr.File(value=None),
				in_samplerate : gr.Textbox(value=None),
				run_btn : gr.Button(interactive=False),
				cancel_btn : gr.Button(interactive=False),
				batch_md : gr.Markdown("", visible=False),
				out_data_file : gr.File(value=None, visible=False)}
	
	# +========================================================================================+
	# |                                    step transition                                     |
	# +========================================================================================+
	def init_next_step(stage1_info, channel_info, fillmode, sel_radio, sel_chkbox):
		if stage1_info["errorFlag"] == True:
			stage1_info["errorFlag"] = False
			return {stage1_json : stage1_info}
		
		# =======================================step1-0========================================
		# step1-0 to step1-1
		if stage1_info["state"] == "step1-initializing":
			# match the names
			stage1_info, channel_info, tpl_montage, in_montage = app_utils.match_name(stage1_info)
			# scale the coordinates
			channel_info = app_utils.align_coords(channel_info, tpl_montage, in_montage)
			# generate and save figures of the montages
			filename1 = stage1_info["fileNames"]["originalMontage"]
			filename2 = stage1_info["fileNames"]["mappedMontage"]
			channel_info = app_utils.save_figure(channel_info, tpl_montage, filename1, filename2)
			
			unassigned_num = len(stage1_info["unassignedInput"])
			if unassigned_num == 0:
				md = """
				### Step1-1: Initial Matching and Scaling
				Below is the result of mapping your channels to our template channels based on their names.
				"""
			else:
				md = """
				### Step1-1: Initial Matching and Scaling
				Below is the result of mapping your channels to our template channels based on their names.<br>
				- channels highlighted in red are those that do not match any template channels.
				"""
			stage1_info["state"] = "step1-finished"
			return {stage1_json : stage1_info,
					channel_json : channel_info,
					map_btn : gr.Button(interactive=True),
					desc_md : gr.Markdown(md),
					tpl_img : gr.Image(visible=True),
					mapped_img : gr.Image(value=filename2, visible=True),
					next_btn : gr.Button(visible=True)}
		
		# =======================================step1-1========================================
		elif stage1_info["state"] == "step1-finished":
			in_num = len(channel_info["inputNames"])
			matched_num = 30 - len(stage1_info["emptyTemplate"])
			
			# step1-1 to step1-4
			if matched_num == 30:
				md = """
				### Mapping Result
				The mapping process has been finished.<br>
				Download the file below if you plan to run the models using the source code.
				"""
				# finalize and save the mapping result
				outputname = stage1_info["fileNames"]["outputData"]
				stage1_info, channel_info = app_utils.mapping_result(stage1_info, channel_info, outputname)
				
				stage1_info["state"] = "finished"
				return {stage1_json : stage1_info,
						channel_json : channel_info,
						desc_md : gr.Markdown(md),
						out_result_file : gr.File(outputname, visible=True),
						tpl_img : gr.Image(visible=False),
						mapped_img : gr.Image(visible=False),
						next_btn : gr.Button(visible=False),
						run_btn : gr.Button(interactive=True)}
			# step1-1 to step1-2
			elif in_num > matched_num: 
				md = """
				### Step1-2: Forwarding Unmatched Channels
				Select one of your unmatched channels to forward its data to the empty template channel 
				currently indicated in red.
				"""
				# initialize the progress indication label
				stage1_info["step2"] = {
					"count" : 1,
					"totalNum" : len(stage1_info["emptyTemplate"])
				}
				tpl_name = stage1_info["emptyTemplate"][0]
				label = '{} (1/{})'.format(tpl_name, stage1_info["step2"]["totalNum"])
				
				stage1_info["state"] = "step2-selecting"
				# determine which button to display
				if stage1_info["step2"]["totalNum"] == 1:
					return {stage1_json : stage1_info,
							desc_md : gr.Markdown(md),
							tpl_img : gr.Image(visible=False),
							mapped_img : gr.Image(visible=False),
							radio_group : gr.Radio(choices=stage1_info["unassignedInput"], value=[], label=label, visible=True),
							clear_btn : gr.Button(visible=True)}
				else:
					return {stage1_json : stage1_info,
							desc_md : gr.Markdown(md),
							tpl_img : gr.Image(visible=False),
							mapped_img : gr.Image(visible=False),
							radio_group : gr.Radio(choices=stage1_info["unassignedInput"], value=[], label=label, visible=True),
							clear_btn : gr.Button(visible=True),
							step2_btn : gr.Button(visible=True),
							next_btn : gr.Button(visible=False)}
			# step1-1 to step1-3-1
			elif in_num == matched_num:
				md = """
				### Step1-3: Filling Remaining Template Channels
				Select one of the methods provided below to fill the remaining template channels.
				"""
				stage1_info["state"] = "step3-select-method"
				return {stage1_json : stage1_info,
						desc_md : gr.Markdown(md),
						tpl_img : gr.Image(visible=False),
						mapped_img : gr.Image(visible=False),
						in_fillmode : gr.Dropdown(visible=True),
						fillmode_btn : gr.Button(visible=True),
						next_btn : gr.Button(visible=False)}
		
		# =======================================step1-2========================================
		elif stage1_info["state"] == "step2-selecting":
			
			if sel_radio != []:
				stage1_info["unassignedInput"].remove(sel_radio)
				
				prev_tpl_name = stage1_info["emptyTemplate"][stage1_info["step2"]["count"]-1]
				prev_tpl_idx = channel_info["templateDict"][prev_tpl_name]["index"]
				sel_idx = channel_info["inputDict"][sel_radio]["index"]
				
				stage1_info["mappingResult"][0]["index"][prev_tpl_idx] = [sel_idx]
				stage1_info["mappingResult"][0]["isOriginalData"][prev_tpl_idx] = True
				channel_info["templateDict"][prev_tpl_name]["matched"] = True
				channel_info["inputDict"][sel_radio]["assigned"] = True
			
			# exclude the tpl_channels filled in step1-2
			stage1_info["emptyTemplate"] = app_utils.get_empty_template(channel_info["templateNames"],
																			channel_info["templateDict"])
			
			# step1-2 to step1-4
			if len(stage1_info["emptyTemplate"]) == 0:
				md = """
				### Mapping Result
				The mapping process has been finished.<br>
				Download the file below if you plan to run the models using the source code.
				"""
				outputname = stage1_info["fileNames"]["outputData"]
				stage1_info, channel_info = app_utils.mapping_result(stage1_info, channel_info, outputname)
				
				stage1_info["state"] = "finished"
				return {stage1_json : stage1_info,
						channel_json : channel_info,
						desc_md : gr.Markdown(md),
						out_result_file : gr.File(outputname, visible=True),
						radio_group : gr.Radio(visible=False),
						clear_btn : gr.Button(visible=False),
						next_btn : gr.Button(visible=False),
						run_btn : gr.Button(interactive=True)}
			# step1-2 to step1-3-1
			else:
				md = """
				### Step1-3: Filling Remaining Template Channels
				Select one of the methods provided below to fill the remaining template channels.
				"""
				stage1_info["state"] = "step3-select-method"
				return {stage1_json : stage1_info,
						channel_json : channel_info,
						desc_md : gr.Markdown(md),
						radio_group : gr.Radio(visible=False),
						in_fillmode : gr.Dropdown(visible=True),
						fillmode_btn : gr.Button(visible=True),
						clear_btn : gr.Button(visible=False),
						next_btn : gr.Button(visible=False)}
		
		# ======================================step1-3-1=======================================
		elif stage1_info["state"] == "step3-select-method":
			# step1-3-1 to step1-4
			if fillmode == "zero":
				md = """
				### Mapping Result
				The mapping process has been finished.<br>
				Download the file below if you plan to run the models using the source code.
				"""
				outputname = stage1_info["fileNames"]["outputData"]
				stage1_info, channel_info = app_utils.mapping_result(stage1_info, channel_info, outputname)
				
				stage1_info["state"] = "finished"
				return {stage1_json : stage1_info,
						channel_json : channel_info,
						desc_md : gr.Markdown(md),
						out_result_file : gr.File(outputname, visible=True),
						in_fillmode : gr.Dropdown(visible=False),
						fillmode_btn : gr.Button(visible=False),
						run_btn : gr.Button(interactive=True)}
			# step1-3-1 to step1-3-2
			elif fillmode == "mean":
				md = """
				### Step1-3: Fill the remaining template channels
				The current empty template channel, indicated in red, will be filled with the average 
				value of the data from the selected channels. (By default, the 4 nearest channels are pre-selected.)
				"""
				# find the 4 nearest in_channels for each unmatched tpl_channel
				stage1_info["mappingResult"][0]["index"] = app_utils.find_neighbors(
																		channel_info,
																		stage1_info["emptyTemplate"],
																		stage1_info["mappingResult"][0]["index"])
				# initialize the progress indication label
				stage1_info["step3"] = {
					"count" : 1,
					"totalNum" : len(stage1_info["emptyTemplate"])
				}
				tpl_name = stage1_info["emptyTemplate"][0]
				label = '{} (1/{})'.format(tpl_name, stage1_info["step3"]["totalNum"])
				
				tpl_idx = channel_info["templateDict"][tpl_name]["index"]
				value = stage1_info["mappingResult"][0]["index"][tpl_idx]
				value = [channel_info["inputNames"][i] for i in value]
				
				stage1_info["state"] = "step3-2-selecting"
				# determine which button to display
				if stage1_info["step3"]["totalNum"] == 1:
					return {stage1_json : stage1_info,
							desc_md : gr.Markdown(md),
							in_fillmode : gr.Dropdown(visible=False),
							fillmode_btn : gr.Button(visible=False),
							chkbox_group : gr.CheckboxGroup(choices=channel_info["inputNames"],
															value=value, label=label, visible=True),
							next_btn : gr.Button(visible=True)}
				else:
					return {stage1_json : stage1_info,
							desc_md : gr.Markdown(md),
							in_fillmode : gr.Dropdown(visible=False),
							fillmode_btn : gr.Button(visible=False),
							chkbox_group : gr.CheckboxGroup(choices=channel_info["inputNames"],
															value=value, label=label, visible=True),
							step3_btn : gr.Button(visible=True)}
		
		# ======================================step1-3-2=======================================
		# step1-3-2 to step1-4
		elif stage1_info["state"] == "step3-2-selecting":
			
			prev_tpl_name = stage1_info["emptyTemplate"][stage1_info["step3"]["count"]-1]
			prev_tpl_idx = channel_info["templateDict"][prev_tpl_name]["index"]
			sel_idx = [channel_info["inputDict"][name]["index"] for name in sel_chkbox]
			stage1_info["mappingResult"][0]["index"][prev_tpl_idx] = sel_idx if sel_idx!=[] else [None]
			
			md = """
			### Mapping Result
			The mapping process has been finished.<br>
			Download the file below if you plan to run the models using the source code.
			"""
			outputname = stage1_info["fileNames"]["outputData"]
			stage1_info, channel_info = app_utils.mapping_result(stage1_info, channel_info, outputname)
			
			stage1_info["state"] = "finished"
			return {stage1_json : stage1_info,
					channel_json : channel_info,
					desc_md : gr.Markdown(md),
					out_result_file : gr.File(outputname, visible=True),
					chkbox_group : gr.CheckboxGroup(visible=False),
					next_btn : gr.Button(visible=False),
					run_btn : gr.Button(interactive=True)}
	
	next_btn.click(
		fn = init_next_step,
		inputs = [stage1_json, channel_json, in_fillmode, radio_group, chkbox_group],
		outputs = [stage1_json, channel_json, desc_md, out_result_file, tpl_img, mapped_img, radio_group,
					in_fillmode, fillmode_btn, chkbox_group, clear_btn, step2_btn, step3_btn, next_btn, run_btn]
	).success(
		fn = None,
		js = init_js,
		inputs = [stage1_json, channel_json],
		outputs = []
	)
	
	# +========================================================================================+
	# |                                      Stage1-step0                                      |
	# +========================================================================================+
	map_btn.click(
		fn = reset_all,
		inputs = [session_dir, stage1_json, stage2_json, in_loc_file],
		outputs = [stage1_json, stage2_json, channel_json, map_btn, desc_md, out_result_file, tpl_img, mapped_img,
					radio_group, in_fillmode, fillmode_btn, chkbox_group, clear_btn, step2_btn, step3_btn, next_btn,
					in_data_file, in_samplerate, run_btn, cancel_btn, batch_md, out_data_file]
	).success(
		fn = init_next_step,
		inputs = [stage1_json, channel_json, in_fillmode, radio_group, chkbox_group], 
		outputs = [stage1_json, channel_json, map_btn, desc_md, tpl_img, mapped_img, next_btn]
	)
	
	# +========================================================================================+
	# |                                      Stage1-step2                                      |
	# +========================================================================================+
	@radio_group.select(inputs = stage1_json, outputs = [step2_btn, next_btn])
	def determine_button(stage1_info):
		if len(stage1_info["unassignedInput"]) == 1:
			return {step2_btn : gr.Button(visible=False),
					next_btn : gr.Button(visible=True)}
		else:
			return {step2_btn : gr.Button()}
	# clear the selected value and reset the buttons
	@clear_btn.click(inputs = stage1_json, outputs = [radio_group, step2_btn, next_btn])
	def clear_value(stage1_info):
		if len(stage1_info["unassignedInput"])==1 and stage1_info["step2"]["count"]<stage1_info["step2"]["totalNum"]:
			return {radio_group : gr.Radio(value=[]),
					step2_btn : gr.Button(visible=True),
					next_btn : gr.Button(visible=False)}
		else:
			return {radio_group : gr.Radio(value=[])}
	
	def update_radio(stage1_info, channel_info, sel_name):
		step2 = stage1_info["step2"]
		# check if the user has selected an in_channel to forward to the previous target tpl_channel
		if sel_name != []:
			stage1_info["unassignedInput"].remove(sel_name)
			
			prev_tpl_name = stage1_info["emptyTemplate"][step2["count"]-1]
			prev_tpl_idx = channel_info["templateDict"][prev_tpl_name]["index"]
			sel_idx = channel_info["inputDict"][sel_name]["index"]
			
			stage1_info["mappingResult"][0]["index"][prev_tpl_idx] = [sel_idx]
			stage1_info["mappingResult"][0]["isOriginalData"][prev_tpl_idx] = True
			channel_info["templateDict"][prev_tpl_name]["matched"] = True
			channel_info["inputDict"][sel_name]["assigned"] = True
		
		# update the new round
		step2["count"] += 1
		tpl_name = stage1_info["emptyTemplate"][step2["count"]-1]
		label = '{} ({}/{})'.format(tpl_name, step2["count"], step2["totalNum"])
		
		stage1_info["step2"] = step2
		# determine which button to display
		if step2["count"] == step2["totalNum"]:
			return {stage1_json : stage1_info,
					channel_json : channel_info,
					radio_group : gr.Radio(choices=stage1_info["unassignedInput"],
											value=[], label=label),
					step2_btn : gr.Button(visible=False),
					next_btn : gr.Button(visible=True)}
		else:
			return {stage1_json : stage1_info,
					channel_json : channel_info,
					radio_group : gr.Radio(choices=stage1_info["unassignedInput"],
											value=[], label=label)}
	step2_btn.click(
		fn = update_radio,
		inputs = [stage1_json, channel_json, radio_group],
		outputs = [stage1_json, channel_json, radio_group, step2_btn, next_btn]
	).success(
		fn = None,
		js = update_js,
		inputs = [stage1_json, channel_json],
		outputs = []
	)
	
	# +========================================================================================+
	# |                                      Stage1-step3                                      |
	# +========================================================================================+	
	def update_chkbox(stage1_info, channel_info, sel_name):
		step3 = stage1_info["step3"]
		
		prev_tpl_name = stage1_info["emptyTemplate"][step3["count"]-1]
		prev_tpl_idx = channel_info["templateDict"][prev_tpl_name]["index"]
		sel_idx = [channel_info["inputDict"][name]["index"] for name in sel_name]
		stage1_info["mappingResult"][0]["index"][prev_tpl_idx] = sel_idx if sel_idx!=[] else [None]
		
		# update the new round
		step3["count"] += 1
		tpl_name = stage1_info["emptyTemplate"][step3["count"]-1]
		label = '{} ({}/{})'.format(tpl_name, step3["count"], step3["totalNum"])
		
		tpl_idx = channel_info["templateDict"][tpl_name]["index"]
		value = stage1_info["mappingResult"][0]["index"][tpl_idx]
		value = [channel_info["inputNames"][i] for i in value]
		
		stage1_info["step3"] = step3
		# determine which button to display
		if step3["count"] == step3["totalNum"]:
			return {stage1_json : stage1_info,
					chkbox_group : gr.CheckboxGroup(value=value, label=label),
					step3_btn : gr.Button(visible=False),
					next_btn : gr.Button(visible=True)}
		else:
			return {stage1_json : stage1_info,
					chkbox_group : gr.CheckboxGroup(value=value, label=label)}
	
	fillmode_btn.click(
		fn = init_next_step,
		inputs = [stage1_json, channel_json, in_fillmode, radio_group, chkbox_group],
		outputs = [stage1_json, channel_json, desc_md, out_result_file, in_fillmode, fillmode_btn,
					chkbox_group, step3_btn, next_btn, run_btn]
	).success(
		fn = None,
		js = init_js,
		inputs = [stage1_json, channel_json],
		outputs = []
	)
	
	step3_btn.click(
		fn = update_chkbox,
		inputs = [stage1_json, channel_json, chkbox_group],
		outputs = [stage1_json, chkbox_group, step3_btn, next_btn]
	).success(
		fn = None,
		js = update_js,
		inputs = [stage1_json, channel_json],
		outputs = []
	)
	
	# +========================================================================================+
	# |                                  Stage2: data denoising                                |
	# +========================================================================================+
	@cancel_btn.click(inputs = stage2_json, outputs = [stage2_json, cancel_btn, batch_md])
	def stop_stage2(stage2_info):
		utils.dataDelete(stage2_info["filePath"])
		stage2_info["state"] = "stopped"
		return stage2_info, gr.Button(interactive=False), gr.Markdown(visible=False)
	
	def reset_stage2(rootpath, stage2_info, in_data, samplerate, modelname):
		if in_data==None or samplerate=="":
			gr.Warning("Please upload a file and enter the sampling rate.")
			stage2_info["errorFlag"] = True
			return {stage2_json : stage2_info}
		elif samplerate.isdigit() == False:
			gr.Warning("The sampling rate must be an integer.")
			stage2_info["errorFlag"] = True
			return {stage2_json : stage2_info}
		
		# delete the previous folder of Stage2
		if "filePath" in stage2_info and stage2_info.get("state")=="finished":
			utils.dataDelete(stage2_info["filePath"])
		# establish a new folder
		stage2_dir = uuid.uuid4().hex + '_stage2/'
		os.mkdir(rootpath + stage2_dir)
		
		inputname = os.path.basename(str(in_data))
		outputname = modelname + '_'+inputname[:-4] + '.csv'
		
		stage2_info = {
			"filePath" : rootpath + stage2_dir,
			"fileNames" : {
				"inputData" : in_data,
				"outputData" : rootpath + stage2_dir + outputname
			},
			"state" : "running",
			"errorFlag" : False
		}
		return {stage2_json : stage2_info,
				run_btn : gr.Button(visible=False),
				cancel_btn : gr.Button(visible=True, interactive=True),
				batch_md : gr.Markdown("", visible=True),
				out_data_file : gr.File(value=None, visible=False)}
	
	def run_model(stage1_info, stage2_info, channel_info, samplerate, modelname):
		if stage2_info["errorFlag"] == True:
			stage2_info["errorFlag"] = False
			yield {stage2_json : stage2_info}
		
		else:
			filepath = stage2_info["filePath"]
			inputname = stage2_info["fileNames"]["inputData"]
			outputname = stage2_info["fileNames"]["outputData"]
			channel_num = len(channel_info["inputNames"])
			mapping_result = stage1_info["mappingResult"]
			
			break_flag = False
			for i in range(stage1_info["batch"]):
				yield {batch_md : gr.Markdown('Running model({}/{})...'.format(i+1, stage1_info["batch"]))}
				try:
					# step1: Data preprocessing
					preprocess_data = utils.preprocessing(filepath, inputname, int(samplerate), mapping_result[i])
					# step2: Signal reconstruction
					reconstructed_data = utils.reconstruct(modelname, preprocess_data, filepath, i)
					# step3: Data postprocessing
					utils.postprocessing(reconstructed_data, int(samplerate), outputname, mapping_result[i], i, channel_num)
				except FileNotFoundError:
					print('stop!!')
					break_flag = True
					break
			
			if break_flag == False:
				stage2_info["state"] = "finished"
				yield {stage2_json : stage2_info,
						run_btn : gr.Button(visible=True),
						cancel_btn : gr.Button(visible=False),
						batch_md : gr.Markdown(visible=False),
						out_data_file : gr.File(outputname, visible=True)}
			else:
				yield {run_btn : gr.Button(visible=True),
						cancel_btn : gr.Button(visible=False)}
	run_btn.click(
		fn = reset_stage2,
		inputs = [session_dir, stage2_json, in_data_file, in_samplerate, in_modelname],
		outputs = [stage2_json, run_btn, cancel_btn, batch_md, out_data_file]
	).success(
		fn = run_model,
		inputs = [stage1_json, stage2_json, channel_json, in_samplerate, in_modelname],
		outputs = [stage2_json, run_btn, cancel_btn, batch_md, out_data_file]
	)
	
	def delete_dir(req: gr.Request):
		utils.dataDelete(gradio_temp_dir+'/'+req.session_hash)
	demo.unload(delete_dir)
	
if __name__ == "__main__":
	demo.launch()

