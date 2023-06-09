import os
import re
import numpy as np
import copy
import json
import jsonpickle
from datetime import datetime
import algorithm_update
from flask import Flask, render_template, request, redirect, session, url_for, render_template_string
from werkzeug.utils import secure_filename

debug=1
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.secret_key = "secretStuff"
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = 'manifests'
#-------------------------NOTE: Change this to your own path-------------------------#
base="D:/CodeFiles/CS179M/"
#-------------------------NOTE: Change this to your own path-------------------------#

log_file = base+'logfile.log'
def log(append_str):
    with open(log_file, 'a') as f:
        f.write(str(datetime.utcnow()) + ' ' + append_str + '\n')
    f.close()

def get_length(generator):
    if hasattr(generator,"__len__"):
        return len(generator)
    else:
        return sum(1 for _ in generator)


@app.route("/")
def home():
    return render_template("home.html")


# Route for handling the login page logic
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        session['user'] = username
        log(username + " has logged in.")
        return redirect('/dashboard')

    return render_template("login.html")


@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    if 'user' in session:
        return render_template("dashboard.html", username=session['user'])

    return '<h1>You are not logged in.</h1>'


@app.route('/logger', methods=['POST'])
def logger():
    if request.method == 'POST':
        data = request.get_json()
        log("Operator has commented: " + data)
        if 'json_data' in session:
            return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/service', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        if f != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return "Invalid file", 400
            if not os.path.exists('manifests'):
                os.makedirs('manifests')
            filePath = os.path.join(app.config['UPLOAD_PATH'], filename)
            f.save(filePath)
            session['filePath'] = filePath

            log(filename + ' was uploaded to the system.')

            openFile = open(filePath, 'r')
            containers = []
            ship_grid = algorithm_update.create_ship_grid(8, 12)
            algorithm_update.update_ship_grid(openFile, ship_grid, containers)
            ship_grid_pickle = jsonpickle.encode([ship_grid], unpicklable=False)
            ship_grid_json = json.dumps(ship_grid_pickle, indent=4)
            session['ship_grids'] = ship_grid_json

            ship_grid_flipped = ship_grid[::-1][:]
        option = request.form['services']
        log(option + ' was selected by operator.')
        if option == 'Transfer':
            return render_template('initialTransferService.html', ship_grid=ship_grid_flipped, enumerate=enumerate,
                                   len=len)
        else:
            return render_template('initialBalancePage.html', ship_grid=ship_grid_flipped, enumerate=enumerate, len=len)


# Transfer functions and html templates

#--------------------------------------------Write to log file--------------------------------------------#
@app.route('/Writing_Log', methods=['POST'])
def handle_form_submission():
    input_text = request.form['input']
    # Do something with the input text, such as write it to a log file
    print(input_text)
    print(log_file)
    with open(log_file, 'w') as f:
        f.writelines("User added a comment: " + input_text+'\n')
    f.close()

    return render_template_string('''
        <script>
            window.close();
        </script>
    ''')




@app.route('/openlog', methods=['POST'])
def openlog():
    if request.method == 'POST':
        return render_template('Writing_Log.html')
#--------------------------------------------Write to log file--------------------------------------------#
@app.route('/starttransfer', methods=['POST'])
def starttransfer():
    if request.method == 'POST':

        # User inputed contaienrs for unload and grid coordinates for load
        ret = session.get('json_data', None)

        unload_containers = ret['unloading']
        load_coordinations = ret['loading']

        print('unload json', unload_containers)

        container_values = []
        for dictionary in unload_containers:
            container_values.append(list(dictionary.values()))

        load_coords = []
        for dictionary in load_coordinations:
            load_coords.append(dictionary['grid position'])

        # Open manifest to create grid and update it with the proper containers
        filePath = session.get('filePath', None)
        openFile = open(filePath, 'r')
        containers = []
        ship_grid = algorithm_update.create_ship_grid(8, 12)
        openFile = open(filePath, 'r')
        algorithm_update.update_ship_grid(openFile, ship_grid, containers)

        # Finding which containers are we need to unload from the ship
        container_locs_unload = []
        for container in containers:
            # print(container)
            for values in container_values:
                name, weight = values[0], int(values[1])
                # print(name, weight)
                if ship_grid[container[0]][container[1]].container.name == name and \
                        ship_grid[container[0]][container[1]].container.weight == weight:
                    container_locs_unload.append(container)

                    log("Selected Container {} with weight {} for unloading".format(name, weight))

        session['container_locs_unload'] = container_locs_unload
        print('container_locs_unload', container_locs_unload)
        # flip container locations
        # '[7,3]','[7,4]'
        adjusted_load_coords = []
        for idx, container in enumerate(load_coords):
            r, c = [int(val) for val in container.strip('[]').split(',')]
            adjusted_load_coords.append([7 - r, c])

            # loading
        invalid_locs = 0
        for container in adjusted_load_coords:
            r, c = container[0], container[1]
            if r > 0:
                if ship_grid[r - 1][c].available is True:
                    if [r - 1, c] not in adjusted_load_coords:
                        # invalid location
                        return redirect(url_for('error'))

                # ---------------------Remove in TMP file---------------------

            file_path = session.get('filePath', None)
            file_path = base + "manifests/" + file_path[-13:-4]
            if (os.path.exists(file_path + "__TMP.txt")):
                os.remove(file_path + "__TMP.txt")
            # ---------------------Remove in TMP file---------------------

            # ---------------------Write in TMP file---------------------

            file_path = session.get('filePath', None)
            file_path = base + "manifests/" + file_path[-13:-4]
            if (os.path.exists(file_path + "__TMP.txt")):
                os.remove(file_path + "__TMP.txt")

            bship_grids_json = jsonpickle.decode(session.get('ship_grids'))
            bship_grids = json_to_grid(bship_grids_json)

            bupdated_manifest = algorithm_update.update_manifest(bship_grids[-1])

            print("filename is: ", file_path)
            size = get_length(bupdated_manifest)
            cnt = 1;
            with open(file_path + "__TMP.txt", 'w') as f:
                for line in bupdated_manifest:
                    if (cnt == size):
                        f.write(line)
                    else:
                        f.write(line + '\n')
                    cnt = cnt + 1
                f.close()
            # ---------------------Write in TMP file---------------------

        if len(load_coords) > 0:
            session['load_coords'] = load_coords
            return redirect(url_for('load_container_form'))
        else:
            # unloading
            return redirect(url_for('process_transfer'))


@app.route('/jsresponse', methods=['POST'])
def js_response():
    if request.method == 'POST':
        data = request.get_json()
        session['json_data'] = data
        if 'json_data' in session:
            return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
        # return jsonify(status="success", data=data)


@app.route('/error')
def error():
    return render_template('error.html')


@app.route('/loadcontainerform', methods=['GET', 'POST'])
def load_container_form():
    print("loadingContainer----------")
    load_coords = session.get('load_coords', None)
    adjusted_load_coords = []
    for idx, container in enumerate(load_coords):
        r, c = [int(val) for val in container.strip('[]').split(',')]
        adjusted_load_coords.append([7 - r + 1, c + 1])
    return render_template('loadContainerForm.html', enumerate=enumerate, load_coords=load_coords,
                           adjusted_load_coords=adjusted_load_coords)


def json_to_grid(json_string):
    data = json.loads(json_string)
    ship_grids = []
    for grid in data:
        ship_grid = []
        for r, row in enumerate(grid):
            grid_row = []
            for c, slot in enumerate(row):
                if slot['hasContainer'] is True:
                    grid_row.append(algorithm_update.Slot( \
                        algorithm_update.Container( \
                            slot['container']['name'], \
                            slot['container']['weight']), \
                        slot['hasContainer'], \
                        slot['available']))
                # elif slot['hasContainer'] is False and slot['available'] is False:
                #     row.append(algorithm_update.Slot(None, slot['hasContainer'], slot['available']))
                else:
                    grid_row.append(algorithm_update.Slot(None, slot['hasContainer'], slot['available']))
            ship_grid.append(grid_row)
        ship_grids.append(ship_grid)

    return ship_grids


@app.route('/transferprocessing', methods=['GET', 'POST'])
def process_transfer():
    ship_grids_json = jsonpickle.decode(session.get('ship_grids'))
    ship_grids = json_to_grid(ship_grids_json)

    containers_and_locs_load = []
    all_steps = []
    if request.method == 'POST':
        load_coords = session.get('load_coords', None)
        for i in range(len(load_coords)):
            name_form = 'nameform' + str(i)
            nameForm = request.form.get(name_form)
            weight_form = 'weightform' + str(i)
            weightForm = request.form.get(weight_form)
            if int(weightForm) < 0:
                return render_template('error.html')
            r, c = [int(val) for val in load_coords[i].strip('[]').split(',')]
            containers_and_locs_load.append((algorithm_update.Container(nameForm, int(weightForm)), [7 - r, c]))
            log("Selected Container {} with weight {} for loading.".format(nameForm, weightForm))

    # account for increase in total time
    num_containers_to_transfer = 0

    container_locs_unload = session.get('container_locs_unload')

    print('containers:', container_locs_unload)
    if len(ship_grids) > 0:
        if not containers_and_locs_load:
            ship_grid = copy.deepcopy(ship_grids[0])
            all_steps, ship_grids = algorithm_update.unload(container_locs_unload, ship_grid)
            print('num steps:', len(all_steps))
            print('num ship grids:', len(ship_grids))
            num_containers_to_transfer += len(container_locs_unload)
        else:
            ship_grid = copy.deepcopy(ship_grids[0])
            all_steps, ship_grids = algorithm_update.load(containers_and_locs_load, ship_grid)
            ship_grid = copy.deepcopy(ship_grids[-1])
            num_containers_to_transfer += len(containers_and_locs_load)

            if len(container_locs_unload) > 0:
                new_steps, new_ship_grids = algorithm_update.unload(container_locs_unload, ship_grid)

                for steps in new_steps:
                    all_steps.append(steps)

                ship_grids.append(new_ship_grids)
                r, c = np.array(ship_grids[0]).shape
                ship_grids = algorithm_update.reformat_grid_list(ship_grids, r, c)

                num_containers_to_transfer += len(container_locs_unload)

        ship_grid_flipped = ship_grids[0][::-1][:]
        if len(ship_grids) > 1:
            ship_grids = ship_grids[1:]

        ship_grid_pickle = jsonpickle.encode(ship_grids, unpicklable=False)
        ship_grid_json = json.dumps(ship_grid_pickle, indent=4)
        session['ship_grids'] = ship_grid_json

        total_time = len(list(algorithm_update.flatten(copy.deepcopy(all_steps)))) + (2 * num_containers_to_transfer)
        session['total_time'] = total_time
        log("Estimated Total Time for Service: {} minutes.".format(total_time))

        display_list = []
        path_list = []
        adj_path_list = []
        if len(all_steps) > 0:
            for step in all_steps[0]:
                if not step:
                    continue

                numbers = re.findall(r"\d+", step)
                path_coords, display_coords = [], []
                for r, c in zip(numbers[0::2], numbers[1::2]):
                    path_coords.append('[' + str(7 - int(r)) + ', ' + c + ']')
                    display_coords.append('[' + str(int(r) + 1) + ', ' + str(int(c) + 1) + ']')
                path_step = path_coords[0] + " to " + path_coords[1]
                display_step = display_coords[0] + " to " + display_coords[1]

                adj_path_list.append(path_step.split(" to "))
                path_list.append(path_step)
                display_list.append(display_step)

        print('allsteps', len(all_steps))
        next_move_list = all_steps[1:]
        print('next_move_list', len(next_move_list))
        session['next_move_list'] = next_move_list

        log("Container moved from: " + str(display_list[0][0:6]) + " to " + str(display_list[-1][10:]))
        # ---------------------Remove in TMP file---------------------

        file_path = session.get('filePath', None)
        file_path = base + "manifests/" + file_path[-13:-4]
        if (os.path.exists(file_path + "__TMP.txt")):
            os.remove(file_path + "__TMP.txt")
        # ---------------------Remove in TMP file---------------------

        # ---------------------Write in TMP file---------------------

        file_path = session.get('filePath', None)
        file_path = base + "manifests/" + file_path[-13:-4]
        if (os.path.exists(file_path + "__TMP.txt")):
            os.remove(file_path + "__TMP.txt")

        bship_grids_json = jsonpickle.decode(session.get('ship_grids'))
        bship_grids = json_to_grid(bship_grids_json)

        bupdated_manifest = algorithm_update.update_manifest(bship_grids[-1])

        print("filename is: ", file_path)
        size = get_length(bupdated_manifest)
        cnt = 1;
        with open(file_path + "__TMP.txt", 'w') as f:
            for line in bupdated_manifest:
                if (cnt == size):
                    f.write(line)
                else:
                    f.write(line + '\n')
                cnt = cnt + 1
            f.close()
        # ---------------------Write in TMP file---------------------


        return render_template('transferService.html', ship_grid=ship_grid_flipped, enumerate=enumerate, len=len,
                               display_list=display_list, path_list=adj_path_list, next_move_list=next_move_list,
                               total_time=total_time)
    else:
        return "Error: Invalid Session Variables"



def findAllFile(baseadd):
    for root, ds, fs in os.walk(baseadd):
        for f in fs:
            yield f

@app.route('/restore', methods=['GET', 'POST'])
def restore_manifests():
    #print('restore')
    baseadd = base+'manifests/'
    for i in findAllFile(baseadd):
        if i.endswith('__TMP.txt'):
            tmp = i[0:-9]+'.txt'
            os.remove(baseadd + tmp)
            os.rename(baseadd +i, baseadd + tmp)
    return render_template('login.html')


@app.route('/transfersteps', methods=['GET', 'POST'])
def transfer_steps():
    if request.method == 'POST':
        move_list = session.get('next_move_list', None)
        ship_grids_json = jsonpickle.decode(session.get('ship_grids'))
        ship_grids = json_to_grid(ship_grids_json)

        ship_grid = ship_grids[0]
        if len(ship_grids) > 1:
            ship_grids = ship_grids[1:]

        ship_grid_pickle = jsonpickle.encode(ship_grids, unpicklable=False)
        ship_grid_json = json.dumps(ship_grid_pickle, indent=4)
        session['ship_grids'] = ship_grid_json

        algorithm_update.print_grid(ship_grid)
        ship_grid_flipped = ship_grid[::-1][:]

        total_time = session.get('total_time', None)

        if move_list is not None:
            display_list = []
            path_list = []
            adj_path_list = []
            for step in move_list[0]:
                if not step:
                    continue

                numbers = re.findall(r"\d+", step)
                path_coords, display_coords = [], []
                for r, c in zip(numbers[0::2], numbers[1::2]):
                    path_coords.append('[' + str(7 - int(r)) + ', ' + c + ']')
                    display_coords.append('[' + str(int(r) + 1) + ', ' + str(int(c) + 1) + ']')
                path_step = path_coords[0] + " to " + path_coords[1]
                display_step = display_coords[0] + " to " + display_coords[1]

                # print(path_step, display_step)

                # numbers = re.findall(r"[^\[\],\sa-z]",step)
                # print('numbers',numbers)
                # path_numbers = [chr(ord('7')-ord(numbers[0])+ord('0')),numbers[1],chr(ord('7')-ord(numbers[2])+ord('0')),numbers[3]]
                # adjusted_numbers = [chr(ord(num)+1) for num in numbers]
                # path_step = "["+path_numbers[0]+", "+path_numbers[1]+"] to ["+path_numbers[2]+", "+path_numbers[3]+"]"
                # display_step = "["+adjusted_numbers[0]+", "+adjusted_numbers[1]+"] to ["+adjusted_numbers[2]+", "+adjusted_numbers[3]+"]"

                adj_path_list.append(path_step.split(" to "))
                display_list.append(display_step)
                path_list.append(path_step)
            print(move_list[0])
            next_move_list = move_list[1:]
            session['next_move_list'] = next_move_list

            log("Container moved from: " + str(display_list[0][0:6]) + " to " + str(display_list[-1][10:]))



            # ---------------------Remove in TMP file---------------------

            file_path = session.get('filePath', None)
            file_path = base + "manifests/" + file_path[-13:-4]
            if (os.path.exists(file_path + "__TMP.txt")):
                os.remove(file_path + "__TMP.txt")
            # ---------------------Remove in TMP file---------------------
                # ---------------------Write in TMP file---------------------

                file_path = session.get('filePath', None)
                file_path = base + "manifests/" + file_path[-13:-4]
                if (os.path.exists(file_path + "__TMP.txt")):
                    os.remove(file_path + "__TMP.txt")

                bship_grids_json = jsonpickle.decode(session.get('ship_grids'))
                bship_grids = json_to_grid(bship_grids_json)

                bupdated_manifest = algorithm_update.update_manifest(bship_grids[-1])

                print("filename is: ", file_path)
                size = get_length(bupdated_manifest)
                cnt = 1;
                with open(file_path + "__TMP.txt", 'w') as f:
                    for line in bupdated_manifest:
                        if (cnt == size):
                            f.write(line)
                        else:
                            f.write(line + '\n')
                        cnt = cnt + 1
                    f.close()
            # ---------------------Write in TMP file---------------------



            return render_template('transferService.html', ship_grid=ship_grid_flipped, enumerate=enumerate, len=len,
                                   display_list=display_list, path_list=adj_path_list, next_move_list=next_move_list,
                                   total_time=total_time)


@app.route('/transfercomplete', methods=['GET', 'POST'])
def transfered():
    # ---------------------Remove in TMP file---------------------

    file_path = session.get('filePath', None)
    file_path = base + "manifests/" + file_path[-13:-4]
    if (os.path.exists(file_path + "__TMP.txt")):
        os.remove(file_path + "__TMP.txt")
    # ---------------------Remove in TMP file---------------------


    ship_grids_json = jsonpickle.decode(session.get('ship_grids'))
    ship_grids = json_to_grid(ship_grids_json)
    if ship_grids[-1][7][0].hasContainer:
        ship_grids[-1][7][0].container = None
        ship_grids[-1][7][0].hasContainer = False
        ship_grids[-1][7][0].available = True
    updated_manifest = algorithm_update.update_manifest(ship_grids[-1])

    with open(file_path+ "__NEW.txt", 'w') as f:
        for line in updated_manifest:
            f.writelines(line + '\n')
        f.close()

    log("Transfer service completed. Updated Manifest saved to {}".format(
        file_path + "__NEW.txt"))

    return render_template('transfered.html')


# Balance functions and html templates

@app.route('/startbalance', methods=['GET', 'POST'])
def start_balance():
    if request.method == 'POST':
        filePath = session.get('filePath', None)
        openFile = open(filePath, 'r')
        containers = []
        ship_grid = algorithm_update.create_ship_grid(8, 12)
        openFile = open(filePath, 'r')

        algorithm_update.update_ship_grid(openFile, ship_grid, containers)

        ship_grids_json = jsonpickle.decode(session.get('ship_grids'))
        ship_grids = json_to_grid(ship_grids_json)

        move_list, ship_grids, success = algorithm_update.balance(ship_grid, containers)
        session['success'] = success
        if success is True:
            log('Balance is achievable.')
        else:
            log('Balance is not achievable, SIFT will be performed instead.')
#        if(success is True):
 #           total_time=0
  #      else:
        total_time = len(list(algorithm_update.flatten(copy.deepcopy(move_list))))
        log("Estimated Total Time for Service: {} minutes".format(total_time))
        session['total_time'] = total_time

        if(move_list!=None):
            move_length=len(move_list)
        else:
            move_length=0

        if move_length > 0:
            display_list = []
            path_list = []
            adj_path_list = []
            for step in move_list[0]:
                if not step:
                    continue

                numbers = re.findall(r"\d+", step)
                path_coords, display_coords = [], []
                for r, c in zip(numbers[0::2], numbers[1::2]):
                    path_coords.append('[' + str(7 - int(r)) + ', ' + c + ']')
                    display_coords.append('[' + str(int(r) + 1) + ', ' + str(int(c) + 1) + ']')
                path_step = path_coords[0] + " to " + path_coords[1]
                display_step = display_coords[0] + " to " + display_coords[1]

                adj_path_list.append(path_step.split(" to "))
                display_list.append(display_step)
                path_list.append(path_step)

            next_move_list = move_list[1:]
            session['next_move_list'] = next_move_list
            ship_grid_flipped = ship_grids[0][::-1][:]
            if len(ship_grids) > 1:
                ship_grids = ship_grids[1:]
            ship_grid_pickle = jsonpickle.encode(ship_grids, unpicklable=False)
            ship_grid_json = json.dumps(ship_grid_pickle, indent=4)
            session['ship_grids'] = ship_grid_json

            log("Container moved from: " + str(display_list[0][0:6]) + " to " + str(display_list[-1][10:]))

#---------------------Write in TMP file---------------------
            print("begin to write in file in start balance")
            file_path = session.get('filePath', None)
            file_path = base+"manifests/"+file_path[-13:-4]
            if (os.path.exists(file_path + "__TMP.txt")):
                os.remove(file_path + "__TMP.txt")


            bship_grids_json = jsonpickle.decode(session.get('ship_grids'))
            bship_grids = json_to_grid(bship_grids_json)

            bupdated_manifest = algorithm_update.update_manifest(bship_grids[-1])

            print("filename is: ",file_path)
            size=get_length(bupdated_manifest)
            cnt=1;
            with open(file_path + "__TMP.txt", 'w') as f:
                for line in bupdated_manifest:
                    if(cnt==size):
                        f.write(line)
                    else:
                        f.write(line + '\n')
                    cnt=cnt+1
                f.close()
#---------------------Write in TMP file---------------------
            return render_template('balanceService.html', ship_grid=ship_grid_flipped, enumerate=enumerate, len=len,
                                   display_list=display_list, path_list=adj_path_list, next_move_list=next_move_list,
                                   total_time=total_time)
            # else:
        else:
            # ---------------------Remove in TMP file---------------------
            print("begin to write in file in start balance")
            file_path = session.get('filePath', None)
            file_path = base + "manifests/" + file_path[-13:-4]
            if (os.path.exists(file_path + "__TMP.txt")):
                os.remove(file_path + "__TMP.txt")
            # ---------------------Remove in TMP file---------------------

            balance_status = session.get('success', None)
            if balance_status is True:

                log("Balance service completed. Updated Manifest saved to {}".format(
                    file_path+ "__NEW.txt"))
                return render_template('balanced.html')
            else:

                log("SIFT service completed. Updated Manifest saved to {}".format(
                    file_path + "__NEW.txt"))
                return render_template('sifted.html')


@app.route('/balancesteps', methods=['GET', 'POST'])
def next_step_balance():
    if request.method == 'POST':
        move_list = session.get('next_move_list', None)

        ship_grids_json = jsonpickle.decode(session.get('ship_grids'))
        ship_grids = json_to_grid(ship_grids_json)

        ship_grid = ship_grids[0]

        if len(ship_grids) > 1:
            ship_grids = ship_grids[1:]

        ship_grid_pickle = jsonpickle.encode(ship_grids, unpicklable=False)
        ship_grid_json = json.dumps(ship_grid_pickle, indent=4)
        session['ship_grids'] = ship_grid_json

        ship_grid_flipped = ship_grid[::-1][:]

        if move_list is not None:
            display_list = []
            path_list = []
            adj_path_list = []
            for step in move_list[0]:
                if not step:
                    continue
                numbers = re.findall(r"\d+", step)
                path_coords, display_coords = [], []
                for r, c in zip(numbers[0::2], numbers[1::2]):
                    path_coords.append('[' + str(7 - int(r)) + ', ' + c + ']')
                    display_coords.append('[' + str(int(r) + 1) + ', ' + str(int(c) + 1) + ']')
                path_step = path_coords[0] + " to " + path_coords[1]
                display_step = display_coords[0] + " to " + display_coords[1]

                adj_path_list.append(path_step.split(" to "))
                display_list.append(display_step)
                path_list.append(path_step)

            next_move_list = move_list[1:]
            session['next_move_list'] = next_move_list
            total_time = session.get('total_time', None)

            log("Container moved from: " + str(display_list[0][0:6]) + " to " + str(display_list[-1][10:]))

            # ---------------------Write in TMP file---------------------
            print("begin to write in file in start balance")
            file_path = session.get('filePath', None)
            file_path = base+"manifests/"+file_path[-13:-4]
            if (os.path.exists(file_path + "__TMP.txt")):
                os.remove(file_path + "__TMP.txt")


            bship_grids_json = jsonpickle.decode(session.get('ship_grids'))
            bship_grids = json_to_grid(bship_grids_json)

            bupdated_manifest = algorithm_update.update_manifest(bship_grids[-1])

            print("filename is: ",file_path)
            size=get_length(bupdated_manifest)
            cnt=1;
            with open(file_path + "__TMP.txt", 'w') as f:
                for line in bupdated_manifest:
                    if(cnt==size):
                        f.write(line)
                    else:
                        f.write(line + '\n')
                    cnt=cnt+1
                f.close()
        # ---------------------Write in TMP file---------------------


        if move_list is not None:
            return render_template('balanceService.html', ship_grid=ship_grid_flipped, enumerate=enumerate, len=len,
                                   display_list=display_list, path_list=adj_path_list, next_move_list=next_move_list,
                                   total_time=total_time)


@app.route('/balanced', methods=['GET', 'POST'])
def balanced():
    # ---------------------Remove in TMP file---------------------

    file_path = session.get('filePath', None)
    file_path = base + "manifests/" + file_path[-13:-4]
    if (os.path.exists(file_path + "__TMP.txt")):
        os.remove(file_path + "__TMP.txt")

    # ---------------------Remove in TMP file---------------------

    ship_grids_json = jsonpickle.decode(session.get('ship_grids'))
    ship_grids = json_to_grid(ship_grids_json)

    updated_manifest = algorithm_update.update_manifest(ship_grids[-1])

    with open(file_path + "__NEW.txt", 'w') as f:
        for line in updated_manifest:
            f.writelines(line + '\n')
        f.close()



    balance_status = session.get('success', None)
    if balance_status is True:

        log("Balance service completed. Updated Manifest saved to {}".format(
            file_path + "__NEW.txt"))
        return render_template('balanced.html')
    else:

        log("SIFT service completed. Updated Manifest saved to {}".format(
            file_path+ "__NEW.txt"))
        return render_template('sifted.html')


@app.route('/logout')
def logout():
    session.pop('user')
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)
