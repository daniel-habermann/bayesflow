import numpy as np

from .graphical_simulator import GraphicalSimulator


def irt_simulator():
    # schools have different exam difficulties
    def sample_school():
        mu_exam_mean = np.random.normal(loc=1.1, scale=0.2)
        sigma_exam_mean = abs(np.random.normal(loc=0, scale=1))

        # hierarchical mu/sigma for the exam difficulty standard deviation (logscale)
        mu_exam_std = np.random.normal(loc=0.5, scale=0.3)
        sigma_exam_std = abs(np.random.normal(loc=0, scale=0.5))

        return dict(
            mu_exam_mean=mu_exam_mean,
            sigma_exam_mean=sigma_exam_mean,
            mu_exam_std=mu_exam_std,
            sigma_exam_std=sigma_exam_std,
        )

    # exams have different question difficulties
    def sample_exam(mu_exam_mean, sigma_exam_mean, mu_exam_std, sigma_exam_std):
        # mean question difficulty for an exam
        exam_mean = np.random.normal(loc=mu_exam_mean, scale=sigma_exam_mean)

        # standard deviation of question difficulty
        log_exam_std = np.random.normal(loc=mu_exam_std, scale=sigma_exam_std)
        exam_std = float(np.exp(log_exam_std))

        return dict(exam_mean=exam_mean, exam_std=exam_std)

    # realizations of individual question difficulties
    def sample_question(exam_mean, exam_std):
        question_difficulty = np.random.normal(loc=exam_mean, scale=exam_std)

        return dict(question_difficulty=question_difficulty)

    # realizations of individual student abilities
    def sample_student():
        student_ability = np.random.normal(loc=0, scale=1)

        return dict(student_ability=student_ability)

    # realizations of individual observations
    def sample_observation(question_difficulty, student_ability):
        theta = np.exp(question_difficulty + student_ability) / (1 + np.exp(question_difficulty + student_ability))

        obs = np.random.binomial(n=1, p=theta)

        return dict(obs=obs)

    def meta_fn():
        return {
            "num_exams": np.random.randint(2, 4),
            "num_questions": np.random.randint(10, 21),
            "num_students": np.random.randint(100, 201),
        }

    simulator = GraphicalSimulator(meta_fn=meta_fn)
    simulator.add_node(
        "schools",
        sample_fn=sample_school,
    )
    simulator.add_node(
        "exams",
        sample_fn=sample_exam,
        reps="num_exams",
    )
    simulator.add_node(
        "questions",
        sample_fn=sample_question,
        reps="num_questions",
    )
    simulator.add_node(
        "students",
        sample_fn=sample_student,
        reps="num_students",
    )

    simulator.add_node("observations", sampling_fn=sample_observation)

    simulator.add_edge("schools", "exams")
    simulator.add_edge("schools", "students")
    simulator.add_edge("exams", "questions")
    simulator.add_edge("questions", "observations")
    simulator.add_edge("students", "observations")

    return simulator


def onelevel_simulator():
    def prior():
        beta = np.random.normal([2, 0], [3, 1])
        sigma = np.random.gamma(1, 1)

        return {"beta": beta, "sigma": sigma}

    def likelihood(beta, sigma, N):
        x = np.random.normal(0, 1, size=N)
        y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)

        return {"x": x, "y": y}

    def meta():
        N = np.random.randint(5, 15)

        return {"N": N}

    simulator = GraphicalSimulator(meta_fn=meta)

    simulator.add_node("prior", sample_fn=prior)
    simulator.add_node("likelihood", sample_fn=likelihood)

    simulator.add_edge("prior", "likelihood")

    return simulator


def twolevel_simulator():
    def sample_hypers():
        hyper_mean = np.random.normal()
        hyper_std = np.abs(np.random.normal())

        return {"hyper_mean": float(hyper_mean), "hyper_std": float(hyper_std)}

    def sample_locals(hyper_mean, hyper_std):
        local_mean = np.random.normal(hyper_mean, hyper_std)

        return {"local_mean": float(local_mean)}

    def sample_shared():
        shared_std = np.abs(np.random.normal())

        return {"shared_std": shared_std}

    def sample_y(local_mean, shared_std):
        y = np.random.normal(local_mean, shared_std)

        return {"y": float(y)}

    simulator = GraphicalSimulator()
    simulator.add_node("hypers", sample_fn=sample_hypers, reps=5)

    simulator.add_node(
        "locals",
        sampling_fn=sample_locals,
        reps=6,
    )
    simulator.add_node("shared", sample_fn=sample_shared, reps=1)
    simulator.add_node("y", sample_fn=sample_y, reps=10)

    simulator.add_edge("hypers", "locals")
    simulator.add_edge("locals", "y")
    simulator.add_edge("shared", "y")

    return simulator


def threelevel_simulator():
    def sample_level_1():
        level_1_mean = np.random.normal()

        return {"level_1_mean": float(level_1_mean)}

    def sample_level_2(level_1_mean):
        level_2_mean = np.random.normal(level_1_mean, 1)

        return {"level_2_mean": float(level_2_mean)}

    def sample_level_3(level_2_mean):
        level_3_mean = np.random.normal(level_2_mean, 1)

        return {"level_3_mean": float(level_3_mean)}

    def sample_shared():
        shared_std = np.abs(np.random.normal())

        return {"shared_std": shared_std}

    def sample_y(level_3_mean, shared_std):
        y = np.random.normal(level_3_mean, shared_std, size=10)

        return {"y": y}

    simulator = GraphicalSimulator()
    simulator.add_node("level1", sample_fn=sample_level_1)
    simulator.add_node(
        "level2",
        sample_fn=sample_level_2,
        reps=10,
    )
    simulator.add_node(
        "level3",
        sample_fn=sample_level_3,
        reps=20,
    )
    simulator.add_node("shared", sample_fn=sample_shared)
    simulator.add_node("y", sample_fn=sample_y, reps=10)

    simulator.add_edge("level1", "level2")
    simulator.add_edge("level2", "level3")
    simulator.add_edge("level3", "y")
    simulator.add_edge("shared", "y")

    return simulator
