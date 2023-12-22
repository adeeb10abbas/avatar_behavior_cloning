import click

from avatar_behavior_cloning.envs.avatar_drake_env import AvatarDrakeEnv

@click.command()
@click.option('--timestep', default=0.001, help='Timestep for simulation')
@click.option('--hand_controller_type', default='impedance', help='Type of hand controller: impedance or pid')
@click.option('--hand_state', default='open', help='State of hand: open or closed')
@click.option('--arm_controller_type', default='pose', help='Type of arm controller: inv_dym, pose, impedance')
@click.option('--motion', default='arm_teleo', help='Type of motion: stable or teleop')
@click.option('--teleop_type', default='keyboard', help='Type of teleop: keyboard or vr')
@click.option('--debug', default=False, help='Debug mode')
@click.option('--plot', default=False, help='Plot mode')

def main(timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot):
  click.echo('Timestep: %s' % timestep)
  click.echo('Hand controller type: %s' % hand_controller_type)
  click.echo('Hand state: %s' % hand_state)
  click.echo('Arm controller type: %s' % arm_controller_type)
  click.echo('Motion: %s' % motion)
  click.echo('Teleop type: %s' % teleop_type)
  click.echo('Debug: %s' % debug)
  click.echo('Plot: %s' % plot)

  avatar_drake_env = AvatarDrakeEnv(timestep, hand_controller_type, hand_state, arm_controller_type, motion, teleop_type, debug, plot)
  avatar_drake_env.simulate()
  avatar_drake_env.close()

if __name__ == "__main__":
  main(standalone_mode=False)