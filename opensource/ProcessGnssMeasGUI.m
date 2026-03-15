function ProcessGnssMeasGUI()
%ProcessGnssMeasGUI - Graphical User Interface for GNSS Measurement Tools
%
% Opens a GUI window that lets you:
%   1. Browse for a GnssLogger log file (directory + filename)
%   2. Optionally enter a known true WGS84 position (lat, lon, alt)
%   3. Click "Run Processing" to compute pseudoranges and a WLS PVT solution
%      and display the standard plots
%
% Usage:
%   ProcessGnssMeasGUI          % open the GUI
%
% The opensource directory must be on the MATLAB path, e.g.:
%   addpath('~/gpstools/opensource');
%
% See also ProcessGnssMeasScript, ReadGnssLogger, ProcessGnssMeas,
%          GpsWlsPvt, PlotPseudoranges, PlotCno, PlotPvt

%Author: auto-generated GUI wrapper
%Open Source code for processing Android GNSS Measurements

%% ---- build the figure ------------------------------------------------
fig = uifigure('Name','GNSS Measurement Tools', ...
    'Position',[100 100 680 460], ...
    'Resize','on');

% Title
uilabel(fig, ...
    'Text','GNSS Measurement Tools', ...
    'FontSize',18,'FontWeight','bold', ...
    'HorizontalAlignment','center', ...
    'Position',[10 415 660 35]);

% ---- Input Files panel ------------------------------------------------
filePanel = uipanel(fig, ...
    'Title','Input Files', ...
    'Position',[10 305 660 105]);

uilabel(filePanel,'Text','Data Directory:','Position',[8 62 110 22]);
hDir = uieditfield(filePanel,'text','Position',[125 62 430 22], ...
    'Placeholder','Path to the folder containing the log file');
uibutton(filePanel,'push','Text','Browse...', ...
    'Position',[562 62 88 22], ...
    'ButtonPushedFcn',@(~,~) cbBrowseDir(hDir));

uilabel(filePanel,'Text','Log File:','Position',[8 28 110 22]);
hFile = uieditfield(filePanel,'text','Position',[125 28 430 22], ...
    'Placeholder','e.g. pseudoranges_log_2016_06_30.txt');
uibutton(filePanel,'push','Text','Browse...', ...
    'Position',[562 28 88 22], ...
    'ButtonPushedFcn',@(~,~) cbBrowseFile(hDir, hFile));

% ---- Parameters panel -------------------------------------------------
paramPanel = uipanel(fig, ...
    'Title','Optional: True WGS84 Position (leave blank to skip)', ...
    'Position',[10 230 660 70]);

uilabel(paramPanel,'Text','Latitude (deg):','Position',[8 18 100 22]);
hLat = uieditfield(paramPanel,'text','Position',[112 18 120 22], ...
    'Placeholder','e.g. 37.4226');

uilabel(paramPanel,'Text','Longitude (deg):','Position',[248 18 110 22]);
hLon = uieditfield(paramPanel,'text','Position',[362 18 120 22], ...
    'Placeholder','e.g. -122.0817');

uilabel(paramPanel,'Text','Altitude (m):','Position',[498 18 90 22]);
hAlt = uieditfield(paramPanel,'text','Position',[590 18 60 22], ...
    'Placeholder','e.g. -28');

% ---- Status / output area (created before Run button so its handle can
%      be captured in the callback closure) ------------------------------
uilabel(fig,'Text','Output:','FontWeight','bold','Position',[10 165 60 20]);
hStatus = uitextarea(fig, ...
    'Position',[10 10 660 152], ...
    'Editable','off', ...
    'FontName','Courier New','FontSize',10, ...
    'Value',{'Ready.'; ...
             'Select a data directory and log file, then click Run Processing.'});

% ---- Run button -------------------------------------------------------
hRunBtn = uibutton(fig,'push', ...
    'Text','▶  Run Processing', ...
    'FontSize',13,'FontWeight','bold', ...
    'Position',[230 188 220 36], ...
    'ButtonPushedFcn', ...
    @(btn,~) cbRun(btn, hDir, hFile, hLat, hLon, hAlt, hStatus));

end % ProcessGnssMeasGUI

%% ---- callbacks -------------------------------------------------------

function cbBrowseDir(hDir)
d = uigetdir(hDir.Value, 'Select the data directory');
if ~isequal(d, 0)
    hDir.Value = d;
end
end

function cbBrowseFile(hDir, hFile)
startPath = hDir.Value;
if isempty(startPath) || ~isfolder(startPath)
    startPath = pwd;
end
[fname, fpath] = uigetfile( ...
    {'*.txt;*.csv;*.log','Log Files (*.txt, *.csv, *.log)'; ...
     '*.*','All Files (*.*)'}, ...
    'Select GnssLogger Log File', startPath);
if ~isequal(fname, 0)
    if isempty(hDir.Value)
        hDir.Value = fpath;
    end
    hFile.Value = fname;
end
end

function cbRun(btn, hDir, hFile, hLat, hLon, hAlt, hStatus)
% Validate inputs
dirName    = strtrim(hDir.Value);
prFileName = strtrim(hFile.Value);

if isempty(dirName)
    uialert(btn.Parent,'Please select a data directory.','Missing Input');
    return
end
if isempty(prFileName)
    uialert(btn.Parent,'Please select a log file.','Missing Input');
    return
end
if ~isfolder(dirName)
    uialert(btn.Parent, ...
        sprintf('Directory not found:\n%s', dirName),'Invalid Directory');
    return
end

btn.Enable = 'off';
appendLog(hStatus, '');
appendLog(hStatus, '=== Starting GNSS Processing ===');
appendLog(hStatus, sprintf('Directory : %s', dirName));
appendLog(hStatus, sprintf('File      : %s', prFileName));

% Optional true position
param.llaTrueDegDegM = [];
latStr = strtrim(hLat.Value);
lonStr = strtrim(hLon.Value);
altStr = strtrim(hAlt.Value);
if ~isempty(latStr) && ~isempty(lonStr) && ~isempty(altStr)
    latVal = str2double(latStr);
    lonVal = str2double(lonStr);
    altVal = str2double(altStr);
    if any(isnan([latVal, lonVal, altVal]))
        uialert(btn.Parent, ...
            'True position values must be numeric.','Invalid Input');
        btn.Enable = 'on';
        return
    end
    param.llaTrueDegDegM = [latVal, lonVal, altVal];
    appendLog(hStatus, sprintf('True LLA  : [%.6f, %.6f, %.1f]', ...
        latVal, lonVal, altVal));
end

try
    %% Read log file
    appendLog(hStatus, '> Reading GNSS log file...');
    dataFilter = SetDataFilter;
    [gnssRaw, ~] = ReadGnssLogger(dirName, prFileName, dataFilter);
    if isempty(gnssRaw)
        appendLog(hStatus, 'ERROR: Could not read log file. Check the file path and format.');
        btn.Enable = 'on';
        return
    end

    %% Fetch ephemeris
    appendLog(hStatus, '> Fetching GPS ephemeris (requires internet access)...');
    fctSeconds = 1e-3 * double(gnssRaw.allRxMillis(end));
    utcTime    = Gps2Utc([], fctSeconds);
    allGpsEph  = GetNasaHourlyEphemeris(utcTime, dirName);
    if isempty(allGpsEph)
        appendLog(hStatus, 'ERROR: Could not retrieve ephemeris. Check internet access or copy the ephemeris file manually.');
        btn.Enable = 'on';
        return
    end

    %% Process measurements
    appendLog(hStatus, '> Computing pseudoranges...');
    [gnssMeas] = ProcessGnssMeas(gnssRaw);

    %% Plots – pseudoranges, pseudorange rates, C/N0
    appendLog(hStatus, '> Plotting pseudoranges and C/N0...');
    h1 = figure;  colors = PlotPseudoranges(gnssMeas, prFileName);
    h2 = figure;  PlotPseudorangeRates(gnssMeas, prFileName, colors);
    h3 = figure;  PlotCno(gnssMeas, prFileName, colors);

    %% WLS PVT
    appendLog(hStatus, '> Computing WLS position/velocity solution...');
    gpsPvt = GpsWlsPvt(gnssMeas, allGpsEph);

    %% Plots – PVT
    appendLog(hStatus, '> Plotting PVT results...');
    h4 = figure;
    PlotPvt(gpsPvt, prFileName, param.llaTrueDegDegM, ...
        'Raw Pseudoranges, Weighted Least Squares Solution');
    drawnow;
    h5 = figure;
    PlotPvtStates(gpsPvt, prFileName);

    %% Carrier phase (if present)
    if any(any(isfinite(gnssMeas.AdrM) & gnssMeas.AdrM ~= 0))
        appendLog(hStatus, '> Processing accumulated delta range (carrier phase)...');
        [gnssMeas] = ProcessAdr(gnssMeas);
        h6 = figure; PlotAdr(gnssMeas, prFileName, colors);
        [adrResid] = GpsAdrResiduals(gnssMeas, allGpsEph, param.llaTrueDegDegM);
        drawnow;
        h7 = figure; PlotAdrResids(adrResid, gnssMeas, prFileName, colors);
    end

    appendLog(hStatus, '');
    appendLog(hStatus, '=== Processing complete! Check the generated figures. ===');

catch ME
    appendLog(hStatus, sprintf('ERROR: %s', ME.message));
    appendLog(hStatus, '  (See the MATLAB Command Window for full details.)');
    rethrow(ME);
end

btn.Enable = 'on';
end % cbRun

function appendLog(hStatus, msg)
%appendLog  Append a line to the uitextarea status widget.
current = hStatus.Value;
if ischar(current)
    current = {current};
end
hStatus.Value = [current; {msg}];
scroll(hStatus, 'bottom');
drawnow limitrate;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2016 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
