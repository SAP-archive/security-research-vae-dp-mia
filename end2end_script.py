from datetime import datetime

from tensorflow.keras.backend import clear_session

from core.attack import Attacker
from core.model import ModelContainer
from core.optimization import ModelOptimizer

# ----- CONFIG -----
model_name: str = "InsertModelName"
list_of_attacks: list = ["InsertAttackName"]

run_optimization: bool = True
run_training: bool = True
run_evaluation: bool = True
run_checkpoint_evaluation: bool = True

verbosity_level: int = 0

tax_attack_special_handling = False
tax_attack_configs = []
# special handling because of tf1 ...
for attack_config in list_of_attacks:
    if "tax_attack" in attack_config:
        tax_attack_special_handling = True
        list_of_attacks.remove(attack_config)
        tax_attack_configs.append(attack_config)


# ----- HELPER -----
def log(*args, sep: str = " "):
    now = datetime.now()
    print(f"{str(now)} :", *args, sep=sep)


# ----- OPTIMIZE -----
if run_optimization:
    log("-" * 5, f"run optimization", "-" * 5, sep=" ")
    mdl_opt = ModelOptimizer(model_name)
    mdl_opt.prepare_optimization()
    # mdl_opt.optimize()
    # mdl_opt.optimize_as_data_generator_for_vgg()
    # mdl_opt.optimize_as_data_generator_for_harcnn()

    del mdl_opt
    clear_session()

# ----- TRAIN -----
mc = ModelContainer.create(model_name, verbose=verbosity_level)
mc.load_data()
mc.create_model()

if run_training:
    log("-" * 5, f"run training", "-" * 5, sep=" ")
    mc.train_model()
    mc.save_model()
elif run_evaluation:
    mc.load_model()
else:
    mc = None

# ----- EVALUATION -----
if run_evaluation:
    log("-" * 5, "run evaluation", "-" * 5, sep=" ")
    mc.plot_model_history()
    mc.evaluate_loss(train=True, val=True)
    # mc.evaluate_image_reconstruction(train=True, val=True)
    # mc.evaluate_ssim(train=True, val=True)
    # mc.evaluate_generated_images()
    # mc.evaluate_gradient_file()
    # mc.evaluate_against_vgg16()
    # mc.evaluate_against_harcnn()

    # mc.data.plot_overall_label_histogram()
    # mc.data.plot_label_histogram(mc.get_figure_dir())

    # mc.perturb_own_dataset()


del mc
clear_session()

# ----- ATTACK -----
attacker = Attacker(model_name, verbose=verbosity_level)
attacker.prepare_attack()

for attack_config in list_of_attacks:
    log("-" * 5, f"attack with config: {attack_config}", "-" * 5, sep=" ")
    attacker.set_attack_config(attack_config)
    attacker.perform_attack()

# ----- CHECKPOINTS -----
if run_checkpoint_evaluation:
    mc = ModelContainer.create(model_name, verbose=verbosity_level)
    mc.load_data()
    mc.create_model(trained=True)

    list_of_checkpoints = mc.get_all_checkpoints()

    if run_evaluation:
        for chkpt in list_of_checkpoints:
            log("-" * 5, f"run evaluation for {chkpt}", "-" * 5, sep=" ")
            mc.load_checkpoint_at(chkpt)
            mc.evaluate_loss(train=True, val=True)
            # mc.evaluate_image_reconstruction(train=True, val=True)
            # mc.evaluate_ssim(train=True, val=True)
            # mc.evaluate_generated_images()
            # mc.evaluate_against_vgg16()
            # mc.evaluate_against_harcnn()

    del mc
    clear_session()

    # ----- ATTACK -----
    attacker = Attacker(model_name, verbose=verbosity_level)
    for chkpt in list_of_checkpoints:
        attacker.prepare_attack(use_checkpoint=chkpt)

        for attack_config in list_of_attacks:
            log(
                "-" * 5,
                f"attack {chkpt} with config: {attack_config}",
                "-" * 5,
                sep=" ",
            )
            attacker.set_attack_config(attack_config)
            attacker.perform_attack()

if tax_attack_special_handling:
    attacker = Attacker(model_name, verbose=verbosity_level)
    attacker.prepare_attack()

    for attack_config in tax_attack_configs:
        log("-" * 5, f"attack with config {attack_config}", "-" * 5, sep=" ")
        attacker.set_attack_config(attack_config)
        attacker.perform_attack()

    del attacker
    clear_session()

    if run_checkpoint_evaluation:
        for chkpt in list_of_checkpoints:
            attacker = Attacker(model_name, verbose=verbosity_level)
            attacker.prepare_attack(use_checkpoint=chkpt)
            for attack_config in tax_attack_configs:
                log(
                    "-" * 5,
                    f"attack {chkpt} with config: {attack_config}",
                    "-" * 5,
                    sep=" ",
                )
                attacker.set_attack_config(attack_config)
                attacker.perform_attack()

            del attacker
            clear_session()
