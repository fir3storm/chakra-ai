# chakra/persona.py
"""
Expert Persona Management System for Kimi K3 Runtime on Windows.
Author & Creator: Abhirup Guha (Info Security Solution)

Provides PersonaManager to manage domain expert personas ('infosec', 'architect',
'devops', 'fullstack'), perform dynamic system prompt adaptation, and support
persona listing and switching across multi-agent workflows.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional


class PersonaManager:
    """
    PersonaManager: Manages expert personas ('infosec', 'architect', 'devops', 'fullstack'),
    dynamic system prompt adaptation, and persona listing/switching.
    """

    DEFAULT_PERSONAS: Dict[str, Dict[str, Any]] = {
        "infosec": {
            "role": "infosec",
            "name": "InfoSec Expert & Security Auditor",
            "description": "Specialist in OWASP top 10 vulnerabilities, threat modeling, AST security code audits, cryptographic hardening, and defensive architecture.",
            "system_prompt": (
                "You are the InfoSec Security Audit & Threat Modeling Expert for Kimi K3 PyTorch Engine on Windows.\n"
                "Your primary focus is application security, identifying static AST & OWASP vulnerabilities "
                "(hardcoded credentials, SQL injection, unsafe eval/exec, command injection, weak cryptography), "
                "remediating security flaws, and enforcing zero-trust defensive coding standards.\n"
                "Always analyze code with security rigor, provide score assessments, and offer safe code fixes."
            ),
        },
        "architect": {
            "role": "architect",
            "name": "System Architect & Software Designer",
            "description": "Specialist in high-level system architecture, modular design blueprints, design patterns, separation of concerns, and clean APIs.",
            "system_prompt": (
                "You are the System Architect Expert for Kimi K3 PyTorch Engine on Windows.\n"
                "Your primary focus is high-level software system design, modular architecture decomposition, "
                "enforcing SOLID design principles, clean component abstractions, maintainability, and resource-efficient execution "
                "under hardware constraints (e.g. 8GB RAM).\n"
                "Always design clear, modular blueprints and maintain clear separation of responsibilities across modules."
            ),
        },
        "devops": {
            "role": "devops",
            "name": "DevOps & Infrastructure Engineer",
            "description": "Specialist in Windows/Linux automation, CI/CD pipelines, containerization, environment configuration, and runtime monitoring.",
            "system_prompt": (
                "You are the DevOps & Infrastructure Engineering Expert for Kimi K3 PyTorch Engine on Windows.\n"
                "Your primary focus is build automation, continuous integration, dependency management, PowerShell/Bash scripting, "
                "runtime environment configuration, error logging, and performance monitoring.\n"
                "Always provide resilient, production-ready automation scripts and robust deployment configurations."
            ),
        },
        "fullstack": {
            "role": "fullstack",
            "name": "Full-Stack Engineer & Polyglot Developer",
            "description": "Specialist in end-to-end software development, UI/UX interaction, RESTful/GraphQL API design, backend logic, and database schemas.",
            "system_prompt": (
                "You are the Full-Stack Engineering Expert for Kimi K3 PyTorch Engine on Windows.\n"
                "Your primary focus is end-to-end software development, building responsive user interfaces, designing clean APIs, "
                "implementing performant backend logic, and integrating frontend components with backend storage solutions.\n"
                "Always write clean, tested, efficient code across all software tiers."
            ),
        },
    }

    def __init__(
        self,
        initial_persona: str = "infosec",
        custom_personas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize PersonaManager.

        Args:
            initial_persona: Initial role name to activate ('infosec', 'architect', 'devops', 'fullstack').
            custom_personas: Dict of custom persona definitions to register on initialization.
        """
        self._personas: Dict[str, Dict[str, Any]] = copy.deepcopy(self.DEFAULT_PERSONAS)

        if custom_personas:
            for role_key, spec in custom_personas.items():
                r_key = role_key.lower().strip()
                self._personas[r_key] = {
                    "role": r_key,
                    "name": spec.get("name", r_key.title()),
                    "description": spec.get("description", "Custom expert persona"),
                    "system_prompt": spec.get("system_prompt", f"You are acting as {r_key} expert."),
                }

        self._active_role: str = ""
        self.set_persona(initial_persona)

    def set_persona(self, role: str) -> str:
        """
        Switches the active persona role.

        Args:
            role: Key of the persona to set ('infosec', 'architect', 'devops', 'fullstack', or registered custom key).

        Returns:
            Normalized active persona role key.

        Raises:
            ValueError: If the requested role is not registered.
        """
        normalized_role = role.lower().strip()
        if normalized_role not in self._personas:
            available = list(self._personas.keys())
            raise ValueError(
                f"Unknown persona role '{role}'. Available personas: {available}"
            )

        self._active_role = normalized_role
        return self._active_role

    def get_active_persona(self) -> Dict[str, Any]:
        """
        Returns a copy of the currently active persona info dictionary.

        Returns:
            Dict containing 'role', 'name', 'description', and 'system_prompt'.
        """
        return copy.deepcopy(self._personas[self._active_role])

    def list_personas(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a copy of all registered personas keyed by role name.

        Returns:
            Dict mapping role keys to persona info dicts.
        """
        return copy.deepcopy(self._personas)

    def get_system_prompt(self, custom_instructions: Optional[str] = None) -> str:
        """
        Generates and returns the adapted system prompt for the active persona.

        Args:
            custom_instructions: Optional custom instructions or task context to append.

        Returns:
            Complete system prompt string.
        """
        active_persona = self.get_active_persona()
        base_prompt = active_persona["system_prompt"]

        if custom_instructions and custom_instructions.strip():
            return f"{base_prompt}\n\nAdditional Task Directives:\n{custom_instructions.strip()}"
        return base_prompt

    def add_custom_persona(
        self,
        role: str,
        name: str,
        description: str,
        system_prompt: str,
    ) -> None:
        """
        Registers a new custom persona or updates an existing persona definition.

        Args:
            role: Unique role identifier key.
            name: Display name of the persona.
            description: Description of the persona's focus area.
            system_prompt: System prompt instructions for the persona.
        """
        normalized_role = role.lower().strip()
        self._personas[normalized_role] = {
            "role": normalized_role,
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
        }
