import os
import imageio

INTEREST_THRESHOLD = 1.5


class FrameSaver:
    """
    Class to save environment renders
    """

    def __init__(
        self,
        logger,
        eta,
        task_name="mape",
        run_name="default",
        alg_name="maddpg",
        gif_dir="gifs",
        frames_dir="frames",
    ):
        """
        Create FrameSaver object
        :param logger: logger object to extract info from
        :param eta: curiosity weight factor
        :param task_name: name of task
        :param run_name: name of run used for subdirectory names
        :param alg_name: algorithm name
        :param gif_dir: directory to store gifs in
        :param frames_dir: directory to store individual frames in
        """
        self.logger = logger
        self.eta = eta
        self.task_name = task_name
        self.run_name = run_name
        self.alg_name = alg_name
        self.gif_dir = gif_dir
        self.frames_dir = frames_dir

        self.frames = []
        self.episode = 0
        self.step = 0

    def add_frame(self, frame, episode):
        """
        Add frame for episode
        :param frame: rendered frame
        :param episode: number of episode
        """
        if self.episode != episode:
            self.episode = episode
            self.frames = []
            self.step = 0
        self.frames.append(frame)
        self.step += 1

    def save_episode_gif(self):
        """
        Save episode frames as gif gif - Stored in gifs/<alg>/<run>/ep_<ep>.gif
        """
        if not os.path.isdir(self.gif_dir):
            os.mkdir(self.gif_dir)
        alg_dir = os.path.join(self.gif_dir, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        run_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)

        gif_path = os.path.join(run_dir, "ep_%d.gif" % self.episode)
        imageio.mimsave(gif_path, self.frames)

    def save_last_frame(self):
        """
        Save last frame - Stored in frames/<alg>/<run>/<ep>_<step>.gif
        """
        if not os.path.isdir(self.frames_dir):
            os.mkdir(self.frames_dir)
        alg_dir = os.path.join(self.frames_dir, self.alg_name)
        if not os.path.isdir(alg_dir):
            os.mkdir(alg_dir)
        run_dir = os.path.join(alg_dir, self.run_name)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)

        frame_path = os.path.join(run_dir, "%d_%d.png" % (self.episode, self.step))
        imageio.imsave(frame_path, self.frames[-1])

    def is_interesting(self, curiosities):
        """
        Determine if curiosities show interest
        :param curiosities: curiosity bonus for each agent
        :return: flag if one curiosity was interested
        """
        for cur in curiosities:
            if cur / self.eta >= INTEREST_THRESHOLD:
                return True
        return False

    def save_interesting_frame(self, curiosities):
        """
        Save last frame if interesting
        :param curiosities: curiosity bonus for each agent
        :return: flag if last frame was interesting
        """
        interesting = self.is_interesting(curiosities)
        if interesting:
            print(curiosities, "INTERESTING")
            self.save_last_frame()
        return interesting
